# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/reinforce.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k reinforce -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/reinforce --force

"""REINFORCE (Williams, 1992) with reward-to-go and an optional baseline (PyTorch).

Gradient estimator implemented by :func:`reinforce_loss`::

    grad J(theta) ~= (1/T) sum_t  grad log pi_theta(a_t | s_t) * (G_t - b(s_t))

with ``G_t = sum_{k>=0} gamma^k r_{t+k+1}`` (reward-to-go). Subtracting any
action-independent baseline ``b`` leaves the expectation unchanged because
``E_{a ~ pi}[grad log pi(a|s)] = grad sum_a pi(a|s) = grad 1 = 0``.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """MLP ``obs (B, obs_dim) -> action logits (B, A)``."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def rewards_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = sum_{k>=0} gamma^k r_{t+k+1}``; (T,) -> (T,)."""
    raise NotImplementedError('TODO: implement rewards_to_go (see the reference in src/mlbook)')

def reinforce_loss(logp: torch.Tensor, returns: torch.Tensor, baseline: torch.Tensor | None=None) -> torch.Tensor:
    """``-(1/T) sum_t logp_t * (G_t - b_t)``; minimising it ascends ``J``.

    Args:
        logp: (T,) ``log pi(a_t|s_t)`` with gradients attached.
        returns: (T,) reward-to-go, no gradients.
        baseline: (T,) optional, no gradients (detached).
    """
    raise NotImplementedError('TODO: implement reinforce_loss (see the reference in src/mlbook)')

def sample_action(policy: nn.Module, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
    """Draw ``a ~ pi(.|obs)`` for one observation (obs_dim,); return ``(a, log pi(a|obs))``.

    Sampling is done in NumPy from the softmax of the logits: cheaper than building a
    ``torch.distributions.Categorical`` per environment step, and identical in distribution.
    """
    raise NotImplementedError('TODO: implement sample_action (see the reference in src/mlbook)')

def rollout(env, policy: PolicyNetwork, rng: np.random.Generator) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """One episode. Returns ``(obs (T, obs_dim), rewards (T,), actions (T,))``."""
    raise NotImplementedError('TODO: implement rollout (see the reference in src/mlbook)')

def train_reinforce(env, n_episodes: int, gamma: float=0.99, lr: float=0.01, seed: int=0, use_baseline: bool=True) -> tuple[PolicyNetwork, list[float]]:
    """Vanilla policy gradient, one update per episode, with a running-mean return baseline."""
    raise NotImplementedError('TODO: implement train_reinforce (see the reference in src/mlbook)')
