# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/grpo.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k grpo -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/grpo --force

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

def group_relative_advantages(rewards: torch.Tensor, normalize_std: bool=True, eps: float=1e-06) -> torch.Tensor:
    """``A_i = (r_i - mean_group) / (std_group + eps)`` (or without the std, Dr. GRPO style).

    Args:
        rewards: (P, G) rewards for ``G`` samples of each of ``P`` prompts.
    Returns:
        (P, G) advantages; a group with identical rewards gets all-zero advantages.
    """
    raise NotImplementedError('TODO: implement group_relative_advantages (see the reference in src/mlbook)')

def k3_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token ``k3`` estimate of ``KL(pi || pi_ref)``: ``exp(q - p) - (q - p) - 1`` with ``p=logp``, ``q=logp_ref``.

    Unbiased under samples from ``pi`` and always non-negative. Inputs/outputs (B, T-1).
    """
    raise NotImplementedError('TODO: implement k3_kl (see the reference in src/mlbook)')

def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, clip_eps: float=0.2, clip_eps_high: float | None=None, beta: float=0.04, token_level: bool=False) -> torch.Tensor:
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
    raise NotImplementedError('TODO: implement grpo_loss (see the reference in src/mlbook)')

@dataclass
class GRPOStats:
    mean_reward: float
    kl: float
    frac_all_same: float

def train_grpo(policy: nn.Module, ref: nn.Module, prompts: torch.Tensor, reward_fn, eos_id: int, group_size: int=8, steps: int=20, max_new_tokens: int=2, lr: float=0.001, beta: float=0.04, clip_eps: float=0.2, inner_epochs: int=1, normalize_std: bool=True) -> list[GRPOStats]:
    """GRPO loop: for each prompt sample ``G`` responses, score with ``reward_fn``, update.

    ``reward_fn(tokens, response_mask, prompt_index) -> (B,)`` where ``prompt_index`` (B,) maps each
    row back to its prompt (so a verifier can look up the expected answer).
    """
    raise NotImplementedError('TODO: implement train_grpo (see the reference in src/mlbook)')
