# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/behavioral_cloning.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k behavioral_cloning -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/behavioral_cloning --force

"""Behavioural cloning: turn expert ``(s, a)`` pairs into a policy by supervised learning.

Two learners share the ``fit`` / ``act`` interface:

* :class:`TabularPolicy` -- bins the observation and stores an action histogram per
  bin; unseen bins fall back to the most frequent action overall. This is a
  *pessimistic* function class (no generalisation) that makes the covariate-shift
  failure of BC visible on a 2-D toy in a few hundred steps.
* :class:`MLPPolicy` -- a PyTorch classifier trained with cross-entropy, i.e.
  ``max_theta sum_i log pi_theta(a_i | s_i)`` -- the same objective as SFT of a language model.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

def collect_expert_data(env, expert_fn, n_episodes: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the expert; return ``(obs (N, obs_dim), actions (N,))``."""
    raise NotImplementedError('TODO: implement collect_expert_data (see the reference in src/mlbook)')

class TabularPolicy:
    """Histogram classifier over a grid of ``bin_width`` cells in observation space.

    A bin never seen in training returns the *class prior* (the most frequent expert action
    overall) -- what a classifier with uninformative features falls back to. This is the
    pessimistic "no generalisation" learner that makes the covariate-shift argument concrete.
    """

    def __init__(self, n_actions: int, bin_width: float, rng: np.random.Generator) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _bin(self, obs: np.ndarray) -> tuple[int, ...]:
        raise NotImplementedError('TODO: implement _bin (see the reference in src/mlbook)')

    def fit(self, obs: np.ndarray, actions: np.ndarray) -> None:
        """Accumulate action counts per bin from ``obs (N, obs_dim)`` and ``actions (N,)``."""
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError('TODO: implement act (see the reference in src/mlbook)')

    def n_bins(self) -> int:
        raise NotImplementedError('TODO: implement n_bins (see the reference in src/mlbook)')

class MLPPolicy(nn.Module):
    """``obs (B, obs_dim) -> logits (B, A)``; trained with cross-entropy on expert actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError('TODO: implement act (see the reference in src/mlbook)')

def fit_bc_mlp(policy: MLPPolicy, obs: np.ndarray, actions: np.ndarray, n_epochs: int=200, lr: float=0.01) -> float:
    """Full-batch cross-entropy training; returns the final loss.

    ``L = -(1/N) sum_i log softmax(f(s_i))[a_i]`` -- BC is a classification problem.
    """
    raise NotImplementedError('TODO: implement fit_bc_mlp (see the reference in src/mlbook)')
