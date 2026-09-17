# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/ppo_lm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ppo_lm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/ppo_lm --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def shaped_rewards(scores: torch.Tensor, logp_policy: torch.Tensor, logp_ref: torch.Tensor, response_mask: torch.Tensor, beta: float) -> torch.Tensor:
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
    raise NotImplementedError('TODO: implement shaped_rewards (see the reference in src/mlbook)')

def last_response_index(response_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last position where ``response_mask == 1`` per row. (B, T) -> (B,)."""
    raise NotImplementedError('TODO: implement last_response_index (see the reference in src/mlbook)')

def gae(rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, gamma: float=1.0, lam: float=0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over response tokens.

    ``delta_t = R_t + gamma V_{t+1} - V_t``, ``A_t = delta_t + gamma lam A_{t+1}``,
    ``V_{T} = 0`` after the last response token. Returns ``(advantages, returns)`` each (B, T-1)
    with ``returns = advantages + values`` (the value-function targets).

    Args:
        rewards: (B, T-1) per-token rewards (already zero outside the response).
        values: (B, T-1) critic estimates ``V(s_t)`` aligned with ``rewards``.
        response_mask: (B, T-1).
    """
    raise NotImplementedError('TODO: implement gae (see the reference in src/mlbook)')

def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, clip_eps: float) -> tuple[torch.Tensor, float]:
    """``L^CLIP = -mean_t[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]`` with ``r_t = exp(logp_new - logp_old)``.

    All inputs (B, T-1). Returns ``(loss, clip_fraction)``.
    """
    raise NotImplementedError('TODO: implement ppo_clip_loss (see the reference in src/mlbook)')

def value_loss(values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, response_mask: torch.Tensor, clip_eps: float=0.2) -> torch.Tensor:
    """Clipped value loss ``0.5 * max((V - R)^2, (clip(V, V_old +- eps) - R)^2)``. All inputs (B, T-1)."""
    raise NotImplementedError('TODO: implement value_loss (see the reference in src/mlbook)')

def masked_entropy(logits: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Mean policy entropy over response positions. ``logits`` (B, T-1, V), ``mask`` (B, T-1)."""
    raise NotImplementedError('TODO: implement masked_entropy (see the reference in src/mlbook)')

class AdaptiveKLController:
    """InstructGPT-style controller: nudge ``beta`` so measured KL tracks ``target``."""

    def __init__(self, beta: float, target: float, horizon: int=10) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def update(self, kl: float) -> float:
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

@dataclass
class PPOStats:
    reward: float
    kl: float
    clip_frac: float
    value_loss: float
    entropy: float
    beta: float

def collect_rollouts(policy: nn.Module, ref: nn.Module, critic: nn.Module, prompts: torch.Tensor, max_new_tokens: int, eos_id: int) -> dict[str, torch.Tensor]:
    """Sample responses and record everything PPO needs to update later (all detached).

    Returns a dict with ``tokens`` (B, T), ``mask`` (B, T-1), ``logp_old``, ``logp_ref``, ``values_old`` (B, T-1).
    """
    raise NotImplementedError('TODO: implement collect_rollouts (see the reference in src/mlbook)')

def ppo_update(policy: nn.Module, critic: nn.Module, opt: torch.optim.Optimizer, roll: dict[str, torch.Tensor], scores: torch.Tensor, beta: float, clip_eps: float=0.2, ppo_epochs: int=2, vf_coef: float=0.5, ent_coef: float=0.0) -> tuple[float, float, float]:
    """Run ``ppo_epochs`` of clipped-surrogate updates on one batch of rollouts.

    Returns ``(clip_fraction, value_loss, entropy)`` from the last epoch.
    """
    raise NotImplementedError('TODO: implement ppo_update (see the reference in src/mlbook)')

def train_ppo(policy: nn.Module, ref: nn.Module, critic: nn.Module, prompts: torch.Tensor, reward_fn, eos_id: int, steps: int=30, max_new_tokens: int=8, beta: float=0.05, kl_target: float | None=None, lr: float=0.001, clip_eps: float=0.2) -> list[PPOStats]:
    """Outer PPO loop: rollouts -> reward -> update, ``steps`` times.

    ``reward_fn(tokens, response_mask) -> (B,)`` is the sequence-level reward. If ``kl_target``
    is given an :class:`AdaptiveKLController` adjusts ``beta``; otherwise ``beta`` is fixed.
    """
    raise NotImplementedError('TODO: implement train_ppo (see the reference in src/mlbook)')
