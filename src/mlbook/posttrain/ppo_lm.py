"""PPO for a language model: rewards with a per-token KL penalty, GAE, the clipped
surrogate, a clipped value loss, an adaptive KL controller and a small training loop.

Four networks take part: the policy ``pi_theta`` (trained), the frozen reference
``pi_ref`` (KL anchor), the critic ``V_psi`` (trained) and a reward function
(a reward model, or here a programmatic reward). Tokens are actions; the state is
the prefix; the sequence-level reward lands on the final response token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig, sample, token_log_probs


class TinyCritic(nn.Module):
    """LM backbone + per-token scalar head: ``forward(tokens) -> V(s_t)`` of shape (B, T)."""

    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        self.backbone = TinyCausalLM(cfg)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.backbone.hidden_states(tokens)  # (B, T, d)
        return self.head(h).squeeze(-1)  # (B, T) value of the prefix ending at t


def shaped_rewards(
    scores: torch.Tensor, logp_policy: torch.Tensor, logp_ref: torch.Tensor, response_mask: torch.Tensor, beta: float
) -> torch.Tensor:
    """Per-token reward ``R_t = -beta (log pi(a_t) - log pi_ref(a_t)) + score * [t is last response token]``.

    Args:
        scores: (B,) sequence-level reward (RM score or programmatic reward).
        logp_policy: (B, T-1) per-token log-probs under the sampling policy.
        logp_ref: (B, T-1) per-token log-probs under the reference model.
        response_mask: (B, T-1) 1.0 on response tokens (aligned with the log-probs).
        beta: KL coefficient.
    Returns:
        (B, T-1) rewards, zero outside the response.
    """
    kl_penalty = -beta * (logp_policy - logp_ref) * response_mask  # (B, T-1)
    last_pos = last_response_index(response_mask)  # (B,)
    rewards = kl_penalty.clone()  # (B, T-1)
    rewards[torch.arange(scores.shape[0]), last_pos] += scores  # add the sequence reward at the final token
    return rewards


def last_response_index(response_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last position where ``response_mask == 1`` per row. (B, T) -> (B,)."""
    T = response_mask.shape[1]
    positions = torch.arange(T, device=response_mask.device).unsqueeze(0)  # (1, T)
    return (positions * response_mask).argmax(dim=1)  # (B,)


def gae(
    rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, gamma: float = 1.0, lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over response tokens.

    ``delta_t = R_t + gamma V_{t+1} - V_t``, ``A_t = delta_t + gamma lam A_{t+1}``,
    ``V_{T} = 0`` after the last response token. Returns ``(advantages, returns)`` each (B, T-1)
    with ``returns = advantages + values`` (the value-function targets).

    Args:
        rewards: (B, T-1) per-token rewards (already zero outside the response).
        values: (B, T-1) critic estimates ``V(s_t)`` aligned with ``rewards``.
        response_mask: (B, T-1).
    """
    B, L = rewards.shape
    advantages = torch.zeros_like(rewards)  # (B, T-1)
    next_adv = torch.zeros(B, device=rewards.device)  # (B,) A_{t+1}
    next_value = torch.zeros(B, device=rewards.device)  # (B,) V_{t+1}
    for t in reversed(range(L)):
        m = response_mask[:, t]  # (B,)
        delta = rewards[:, t] + gamma * next_value - values[:, t]  # (B,)
        next_adv = (delta + gamma * lam * next_adv) * m  # (B,) zero outside the response
        advantages[:, t] = next_adv
        next_value = values[:, t] * m  # (B,) V_{t} becomes V_{t+1} for the previous step
    returns = advantages + values  # (B, T-1)
    return advantages, returns


def ppo_clip_loss(
    logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, clip_eps: float
) -> tuple[torch.Tensor, float]:
    """``L^CLIP = -mean_t[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]`` with ``r_t = exp(logp_new - logp_old)``.

    All inputs (B, T-1). Returns ``(loss, clip_fraction)``.
    """
    ratio = torch.exp(logp_new - logp_old)  # (B, T-1)
    unclipped = ratio * advantages  # (B, T-1)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages  # (B, T-1)
    per_token = -torch.min(unclipped, clipped)  # (B, T-1)
    loss = (per_token * response_mask).sum() / response_mask.sum().clamp(min=1.0)  # scalar
    clip_frac = float((((ratio - 1.0).abs() > clip_eps).float() * response_mask).sum() / response_mask.sum().clamp(min=1.0))
    return loss, clip_frac


def value_loss(
    values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, response_mask: torch.Tensor, clip_eps: float = 0.2
) -> torch.Tensor:
    """Clipped value loss ``0.5 * max((V - R)^2, (clip(V, V_old +- eps) - R)^2)``. All inputs (B, T-1)."""
    v_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)  # (B, T-1)
    per_token = 0.5 * torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)  # (B, T-1)
    return (per_token * response_mask).sum() / response_mask.sum().clamp(min=1.0)  # scalar


def masked_entropy(logits: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Mean policy entropy over response positions. ``logits`` (B, T-1, V), ``mask`` (B, T-1)."""
    logp = torch.log_softmax(logits, dim=-1)  # (B, T-1, V)
    ent = -(logp.exp() * logp).sum(dim=-1)  # (B, T-1)
    return (ent * response_mask).sum() / response_mask.sum().clamp(min=1.0)  # scalar


class AdaptiveKLController:
    """InstructGPT-style controller: nudge ``beta`` so measured KL tracks ``target``."""

    def __init__(self, beta: float, target: float, horizon: int = 10) -> None:
        self.beta, self.target, self.horizon = beta, target, horizon

    def update(self, kl: float) -> float:
        error = max(-0.2, min(0.2, kl / self.target - 1.0))
        self.beta *= 1.0 + error / self.horizon
        return self.beta


@dataclass
class PPOStats:
    reward: float
    kl: float
    clip_frac: float
    value_loss: float
    entropy: float
    beta: float


def collect_rollouts(
    policy: nn.Module, ref: nn.Module, critic: nn.Module, prompts: torch.Tensor, max_new_tokens: int, eos_id: int
) -> dict[str, torch.Tensor]:
    """Sample responses and record everything PPO needs to update later (all detached).

    Returns a dict with ``tokens`` (B, T), ``mask`` (B, T-1), ``logp_old``, ``logp_ref``, ``values_old`` (B, T-1).
    """
    tokens, full_mask = sample(policy, prompts, max_new_tokens, eos_id)  # (B, T), (B, T)
    mask = full_mask[:, 1:]  # (B, T-1) aligned with log-probs of tokens[:, 1:]
    with torch.no_grad():
        logp_old = token_log_probs(policy(tokens), tokens)  # (B, T-1)
        logp_ref = token_log_probs(ref(tokens), tokens)  # (B, T-1)
        values_old = critic(tokens)[:, :-1]  # (B, T-1) V(s_t) for the state before tokens[:, t+1]
    return {"tokens": tokens, "mask": mask, "logp_old": logp_old, "logp_ref": logp_ref, "values_old": values_old}


def ppo_update(
    policy: nn.Module,
    critic: nn.Module,
    opt: torch.optim.Optimizer,
    roll: dict[str, torch.Tensor],
    scores: torch.Tensor,
    beta: float,
    clip_eps: float = 0.2,
    ppo_epochs: int = 2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
) -> tuple[float, float, float]:
    """Run ``ppo_epochs`` of clipped-surrogate updates on one batch of rollouts.

    Returns ``(clip_fraction, value_loss, entropy)`` from the last epoch.
    """
    rewards = shaped_rewards(scores, roll["logp_old"], roll["logp_ref"], roll["mask"], beta)  # (B, T-1)
    advantages, returns = gae(rewards, roll["values_old"], roll["mask"])  # (B, T-1) each
    m = roll["mask"]
    adv_mean = (advantages * m).sum() / m.sum()
    adv_std = torch.sqrt((((advantages - adv_mean) ** 2) * m).sum() / m.sum()) + 1e-8
    advantages = (advantages - adv_mean) / adv_std  # (B, T-1) whitened
    for _ in range(ppo_epochs):
        logits = policy(roll["tokens"])  # (B, T, V)
        logp_new = token_log_probs(logits, roll["tokens"])  # (B, T-1)
        pg_loss, clip_frac = ppo_clip_loss(logp_new, roll["logp_old"], advantages, m, clip_eps)
        values = critic(roll["tokens"])[:, :-1]  # (B, T-1)
        v_loss = value_loss(values, roll["values_old"], returns, m)
        ent = masked_entropy(logits[:, :-1, :], m)
        loss = pg_loss + vf_coef * v_loss - ent_coef * ent  # scalar
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 1.0)
        opt.step()
    return clip_frac, float(v_loss.detach()), float(ent.detach())


def train_ppo(
    policy: nn.Module,
    ref: nn.Module,
    critic: nn.Module,
    prompts: torch.Tensor,
    reward_fn,
    eos_id: int,
    steps: int = 30,
    max_new_tokens: int = 8,
    beta: float = 0.05,
    kl_target: float | None = None,
    lr: float = 1e-3,
    clip_eps: float = 0.2,
) -> list[PPOStats]:
    """Outer PPO loop: rollouts -> reward -> update, ``steps`` times.

    ``reward_fn(tokens, response_mask) -> (B,)`` is the sequence-level reward. If ``kl_target``
    is given an :class:`AdaptiveKLController` adjusts ``beta``; otherwise ``beta`` is fixed.
    """
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=lr)
    controller = AdaptiveKLController(beta, kl_target) if kl_target is not None else None
    history: list[PPOStats] = []
    for _ in range(steps):
        roll = collect_rollouts(policy, ref, critic, prompts, max_new_tokens, eos_id)
        full_mask = torch.cat([torch.zeros(prompts.shape[0], 1), roll["mask"]], dim=1)  # (B, T)
        scores = reward_fn(roll["tokens"], full_mask)  # (B,)
        kl = float((((roll["logp_old"] - roll["logp_ref"]) * roll["mask"]).sum(dim=1)).mean())  # per-sequence KL estimate
        clip_frac, v_loss, ent = ppo_update(policy, critic, opt, roll, scores, beta, clip_eps)
        history.append(PPOStats(float(scores.mean()), kl, clip_frac, v_loss, ent, beta))
        if controller is not None:
            beta = controller.update(kl)
    return history
