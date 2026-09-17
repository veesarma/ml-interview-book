"""Canon #60, part 7 -- DPO and a simplified GRPO, each as one small function.

Both algorithms optimise the same KL-regularised objective

``max_pi E_{y ~ pi}[r(x, y)] - beta * KL(pi || pi_ref)``

and they differ in what stands in for ``r``.

DPO never builds a reward model. The optimum of the objective above is
``pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x)``; invert it for ``r``,
substitute into Bradley-Terry, and ``Z(x)`` cancels because it depends only on
``x``. What is left is a classification loss on the difference of two log
ratios, computable with two forward passes of the policy and two of the frozen
reference.

GRPO keeps the RL loop but drops the critic. Sample a group of ``G`` responses
per prompt, score each with the verifier, and use the group-normalised reward as
every token's advantage. The baseline is the group mean, which is what makes the
advantages sum to zero within a group and is why GRPO needs no value network.

Shapes: ``P`` prompts per step, ``G`` samples per prompt, ``B = P * G`` rows.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from mlbook.capstone import synthetic_task as task
from mlbook.capstone.multimodal_model import TinyVLM
from mlbook.capstone.reward_stage import PreferencePair, verifier_rewards
from mlbook.capstone.sft_stage import SFTBatch, encode_batch, next_token_logits


def freeze_reference(model: TinyVLM) -> TinyVLM:
    """Deep-copy a policy and freeze it. The result is ``pi_ref`` for DPO and GRPO."""
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def sequence_logprob(model: TinyVLM, batch: SFTBatch) -> torch.Tensor:
    """Summed log-probability of the assistant tokens. Returns ``(B,)``.

    ``log pi(y | x) = sum_t log pi(y_t | x, y_<t)``, restricted to the positions
    ``loss_mask`` marks. Prompt tokens contribute nothing, which is what makes
    the DPO log ratio a property of the response.
    """
    logits = model.text_logits(batch.images, batch.input_ids, batch.attention_mask)  # (B, T, V)
    logprobs = F.log_softmax(logits, dim=-1)                             # (B, T, V)
    token_logp = logprobs.gather(-1, batch.input_ids[:, :, None])[:, :, 0]  # (B, T)
    return (token_logp * batch.loss_mask).sum(dim=1)                     # (B,)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The DPO objective. All inputs ``(B,)`` summed log-probabilities.

    Returns ``(loss, margin)`` where

    ``margin = (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))``

    is the difference of implicit rewards divided by ``beta``, and

    ``loss = -log sigma(beta * margin)``.

    Training drives ``margin`` up. Watch it rather than the loss: the loss also
    falls when the policy pushes ``log pi(y_l)`` down without raising
    ``log pi(y_w)``, which is the well-known DPO failure of degrading both.
    """
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)  # (B,)
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, margin


@torch.no_grad()
def mean_dpo_margin(policy: TinyVLM, reference: TinyVLM, pairs: list[PreferencePair]) -> float:
    """Mean implicit-reward margin over a set of pairs, in nats."""
    examples = [p.example for p in pairs]
    chosen = encode_batch(examples, answers=[p.chosen for p in pairs])
    rejected = encode_batch(examples, answers=[p.rejected for p in pairs])
    _, margin = dpo_loss(
        sequence_logprob(policy, chosen),
        sequence_logprob(policy, rejected),
        sequence_logprob(reference, chosen),
        sequence_logprob(reference, rejected),
    )
    return float(margin.mean())


def run_dpo(
    policy: TinyVLM,
    reference: TinyVLM,
    pairs: list[PreferencePair],
    steps: int = 120,
    batch_size: int = 32,
    lr: float = 3e-4,
    beta: float = 0.1,
    seed: int = 0,
) -> dict[str, object]:
    """Run DPO on the frozen reference. Returns the loss curve and the margin before/after."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    margin_before = mean_dpo_margin(policy, reference, pairs[:256])
    losses: list[float] = []
    margins: list[float] = []
    policy.train()
    for _ in range(steps):
        idx = rng.integers(0, len(pairs), size=batch_size)
        chunk = [pairs[int(i)] for i in idx]
        examples = [p.example for p in chunk]
        chosen = encode_batch(examples, answers=[p.chosen for p in chunk])
        rejected = encode_batch(examples, answers=[p.rejected for p in chunk])
        with torch.no_grad():
            ref_w = sequence_logprob(reference, chosen)                  # (B,)
            ref_l = sequence_logprob(reference, rejected)                # (B,)
        pol_w = sequence_logprob(policy, chosen)                         # (B,)
        pol_l = sequence_logprob(policy, rejected)                       # (B,)
        loss, margin = dpo_loss(pol_w, pol_l, ref_w, ref_l, beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        margins.append(float(margin.detach().mean()))
    return {
        "losses": losses,
        "margins": margins,
        "margin_before": margin_before,
        "margin_after": mean_dpo_margin(policy, reference, pairs[:256]),
        "loss_first": float(np.mean(losses[:10])),
        "loss_last": float(np.mean(losses[-10:])),
    }


# ---------------------------------------------------------------------------
# GRPO with a verifiable reward
# ---------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, normalize_std: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """``A_i = (r_i - mean_g) / (std_g + eps)``. Input and output ``(P, G)``.

    Subtracting the group mean is the whole trick: it is a baseline that depends
    on the prompt but not on the sampled response, so it leaves the policy
    gradient unbiased while removing the per-prompt difficulty. Every row sums
    to zero by construction, whether or not the std is divided out. Dividing by
    the std reweights easy and hard prompts; Dr. GRPO drops it for that reason.
    """
    centred = rewards - rewards.mean(dim=1, keepdim=True)                # (P, G)
    if not normalize_std:
        return centred                                                   # (P, G)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)               # (P, 1)
    return centred / (std + eps)                                         # (P, G)


def k3_kl(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Schulman's ``k3`` estimator of ``KL(pi || pi_ref)``: ``exp(q - p) - (q - p) - 1``.

    Inputs ``(B,)`` log-probabilities under the policy (``p``) and the reference
    (``q``). Non-negative for every sample and unbiased in expectation, unlike
    the raw ``p - q`` difference which is unbiased but can go negative and makes
    the loss curve unreadable.
    """
    diff = logp_ref - logp_policy                                        # (B,)
    return torch.exp(diff) - diff - 1.0                                  # (B,)


def grpo_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    logp_ref: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
) -> torch.Tensor:
    """Clipped policy-gradient loss with a ``k3`` KL penalty. All inputs ``(B,)``, output scalar.

    ``ratio = exp(log pi - log pi_old)``; the surrogate is
    ``min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)``. On the first inner epoch
    the ratio is exactly 1 and this reduces to REINFORCE with a group baseline.
    """
    ratio = torch.exp(logp - logp_old)                                   # (B,)
    unclipped = ratio * advantages                                       # (B,)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages  # (B,)
    policy_loss = -torch.min(unclipped, clipped).mean()
    kl = k3_kl(logp, logp_ref).mean()
    return policy_loss + kl_coef * kl


@torch.no_grad()
def sample_group(
    model: TinyVLM,
    examples: list[task.Example],
    group_size: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[list[task.Example], list[str], torch.Tensor]:
    """Sample ``G`` answers for each of ``P`` prompts.

    Returns ``(repeated_examples, sampled_answers, rewards)`` with
    ``len == P * G`` and ``rewards`` shaped ``(P, G)``.
    """
    repeated = [ex for ex in examples for _ in range(group_size)]        # P*G
    logits = next_token_logits(model, repeated)                          # (P*G, V)
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)       # (P*G, V)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)[:, 0]  # (P*G,)
    answers = [task.ITOS[int(i)] for i in sampled]
    rewards = verifier_rewards(answers, repeated).reshape(len(examples), group_size)  # (P, G)
    return repeated, answers, rewards


def run_grpo(
    policy: TinyVLM,
    reference: TinyVLM,
    examples: list[task.Example],
    steps: int = 60,
    prompts_per_step: int = 16,
    group_size: int = 8,
    inner_epochs: int = 2,
    lr: float = 2e-4,
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[str, object]:
    """Sample, score with the verifier, normalise within the group, update.

    Each step is one full on-policy iteration: generation, reward, advantage,
    then ``inner_epochs`` gradient steps on the same batch under the clip. That
    ordering is the part worth being able to recite.
    """
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    mean_rewards: list[float] = []
    losses: list[float] = []
    for _ in range(steps):
        idx = rng.integers(0, len(examples), size=prompts_per_step)
        prompts = [examples[int(i)] for i in idx]
        policy.eval()
        repeated, answers, rewards = sample_group(policy, prompts, group_size, temperature, generator)
        mean_rewards.append(float(rewards.mean()))
        advantages = group_advantages(rewards).reshape(-1)               # (P*G,)

        batch = encode_batch(repeated, answers=answers, include_eos=False)
        policy.train()
        with torch.no_grad():
            logp_old = sequence_logprob(policy, batch)                   # (P*G,)
            logp_ref = sequence_logprob(reference, batch)                # (P*G,)
        for _ in range(inner_epochs):
            logp = sequence_logprob(policy, batch)                       # (P*G,)
            loss = grpo_loss(logp, logp_old, advantages, logp_ref, clip_eps, kl_coef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return {
        "losses": losses,
        "mean_rewards": mean_rewards,
        "reward_first": float(np.mean(mean_rewards[:5])),
        "reward_last": float(np.mean(mean_rewards[-5:])),
    }
