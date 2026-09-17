"""Group Relative Policy Optimization (GRPO) with verifiable rewards.

For each prompt sample a *group* of ``G`` responses, score them with a verifier,
and use the group-normalised reward as the advantage of every token in the
response -- no critic. The policy is updated with the PPO ratio clip and a KL
penalty to the reference estimated with the unbiased, non-negative ``k3``
estimator ``exp(q - p) - (q - p) - 1`` where ``p = log pi``, ``q = log pi_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mlbook.posttrain.toy_lm import sample, token_log_probs


def group_relative_advantages(rewards: torch.Tensor, normalize_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """``A_i = (r_i - mean_group) / (std_group + eps)`` (or without the std, Dr. GRPO style).

    Args:
        rewards: (P, G) rewards for ``G`` samples of each of ``P`` prompts.
    Returns:
        (P, G) advantages; a group with identical rewards gets all-zero advantages.
    """
    mean = rewards.mean(dim=1, keepdim=True)  # (P, 1)
    centred = rewards - mean  # (P, G)
    if not normalize_std:
        return centred
    std = rewards.std(dim=1, keepdim=True, unbiased=False)  # (P, 1)
    return centred / (std + eps)  # (P, G)


def k3_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token ``k3`` estimate of ``KL(pi || pi_ref)``: ``exp(q - p) - (q - p) - 1`` with ``p=logp``, ``q=logp_ref``.

    Unbiased under samples from ``pi`` and always non-negative. Inputs/outputs (B, T-1).
    """
    log_ratio = logp_ref - logp  # (B, T-1)
    return torch.exp(log_ratio) - log_ratio - 1.0  # (B, T-1)


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    beta: float = 0.04,
    token_level: bool = False,
) -> torch.Tensor:
    """GRPO surrogate: clipped policy-ratio term with a sequence-level advantage plus ``beta * k3 KL``.

    Args:
        logp_new, logp_old, logp_ref: (B, T-1) per-token log-probs.
        advantages: (B,) one scalar per response, broadcast to its tokens.
        response_mask: (B, T-1).
        clip_eps_high: DAPO "clip-higher": a larger upper bound (defaults to ``clip_eps``).
        token_level: if True average over all response tokens in the batch (DAPO / Dr. GRPO);
            if False average per sequence first, then over sequences (original GRPO).
    Returns:
        scalar loss to minimise.
    """
    hi = clip_eps if clip_eps_high is None else clip_eps_high
    ratio = torch.exp(logp_new - logp_old)  # (B, T-1)
    adv = advantages.unsqueeze(1)  # (B, 1)
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + hi) * adv)  # (B, T-1)
    per_token = -surrogate + beta * k3_kl(logp_new, logp_ref)  # (B, T-1)
    if token_level:
        return (per_token * response_mask).sum() / response_mask.sum().clamp(min=1.0)  # scalar
    per_seq = (per_token * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1.0)  # (B,)
    return per_seq.mean()  # scalar


@dataclass
class GRPOStats:
    mean_reward: float
    kl: float
    frac_all_same: float  # fraction of groups whose rewards were identical (zero gradient)


def train_grpo(
    policy: nn.Module,
    ref: nn.Module,
    prompts: torch.Tensor,
    reward_fn,
    eos_id: int,
    group_size: int = 8,
    steps: int = 20,
    max_new_tokens: int = 2,
    lr: float = 1e-3,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    inner_epochs: int = 1,
    normalize_std: bool = True,
) -> list[GRPOStats]:
    """GRPO loop: for each prompt sample ``G`` responses, score with ``reward_fn``, update.

    ``reward_fn(tokens, response_mask, prompt_index) -> (B,)`` where ``prompt_index`` (B,) maps each
    row back to its prompt (so a verifier can look up the expected answer).
    """
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    P = prompts.shape[0]
    history: list[GRPOStats] = []
    for _ in range(steps):
        prompt_index = torch.arange(P).repeat_interleave(group_size)  # (P*G,)
        batch_prompts = prompts[prompt_index]  # (P*G, T_p)
        tokens, full_mask = sample(policy, batch_prompts, max_new_tokens, eos_id)  # (P*G, T), (P*G, T)
        mask = full_mask[:, 1:]  # (P*G, T-1)
        rewards = reward_fn(tokens, full_mask, prompt_index)  # (P*G,)
        adv = group_relative_advantages(rewards.view(P, group_size), normalize_std).view(-1)  # (P*G,)
        with torch.no_grad():
            logp_old = token_log_probs(policy(tokens), tokens)  # (P*G, T-1)
            logp_ref = token_log_probs(ref(tokens), tokens)  # (P*G, T-1)
        for _ in range(inner_epochs):
            logp_new = token_log_probs(policy(tokens), tokens)  # (P*G, T-1)
            loss = grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps=clip_eps, beta=beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        kl = float(((k3_kl(logp_old, logp_ref) * mask).sum(dim=1)).mean())
        same = float((rewards.view(P, group_size).std(dim=1, unbiased=False) == 0).float().mean())
        history.append(GRPOStats(float(rewards.mean()), kl, same))
    return history
