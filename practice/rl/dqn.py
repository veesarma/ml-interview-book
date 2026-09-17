# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/dqn.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k dqn -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/dqn --force

"""Deep Q-Network (Mnih et al., 2013/2015) with Double DQN and Dueling heads (PyTorch).

Loss on a minibatch of transitions ``(s, a, r, s', d)``::

    y = r + gamma * (1 - d) * max_a' Q_target(s', a')          # DQN
    y = r + gamma * (1 - d) * Q_target(s', argmax_a' Q_online(s', a'))   # Double DQN
    L = Huber( Q_online(s, a) - y )

The target ``y`` is treated as a constant (no gradient flows through the target
network), which is what makes this *semi-gradient* TD rather than a true gradient
of a Bellman residual.

Dimension vocabulary: ``B`` batch, ``obs_dim`` observation width, ``A`` actions.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

class QNetwork(nn.Module):
    """MLP ``obs (B, obs_dim) -> Q-values (B, A)``."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class DuelingQNetwork(nn.Module):
    """Dueling head (Wang et al., 2016): ``Q = V + (A - mean_a A)``.

    Subtracting the mean makes the decomposition identifiable (otherwise any
    constant could be moved between ``V`` and ``A``).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class ReplayBuffer:
    """Ring buffer of transitions; uniform sampling breaks temporal correlation."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def add(self, obs: np.ndarray, a: int, r: float, next_obs: np.ndarray, done: bool) -> None:
        raise NotImplementedError('TODO: implement add (see the reference in src/mlbook)')

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError('TODO: implement sample (see the reference in src/mlbook)')

def dqn_loss(q_net: nn.Module, target_net: nn.Module, batch: tuple[torch.Tensor, ...], gamma: float, double: bool=False) -> torch.Tensor:
    """Huber TD loss on a batch; returns a scalar tensor.

    Args:
        batch: ``(obs (B, obs_dim), actions (B,), rewards (B,), next_obs (B, obs_dim), dones (B,))``.
        double: use the online net to *select* ``a'`` and the target net to *evaluate* it.
    """
    raise NotImplementedError('TODO: implement dqn_loss (see the reference in src/mlbook)')

def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Linear anneal from ``eps_start`` to ``eps_end`` over ``decay_steps`` environment steps."""
    raise NotImplementedError('TODO: implement epsilon_schedule (see the reference in src/mlbook)')

@dataclass
class DQNConfig:
    gamma: float = 0.98
    lr: float = 0.002
    batch_size: int = 32
    buffer_size: int = 20000
    warmup: int = 200
    target_update_every: int = 200
    train_every: int = 3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 1500
    double: bool = False
    hidden: int = 64

def train_dqn(env, n_episodes: int, cfg: DQNConfig, seed: int=0, dueling: bool=False) -> tuple[nn.Module, list[float]]:
    """Run DQN on an episodic env with ``obs_dim`` / ``n_actions`` attributes.

    Returns the trained online network and the list of undiscounted episode returns.
    """
    raise NotImplementedError('TODO: implement train_dqn (see the reference in src/mlbook)')

def greedy_action(q_net: nn.Module, obs: np.ndarray) -> int:
    """``argmax_a Q(obs, a)`` for a single observation (obs_dim,)."""
    raise NotImplementedError('TODO: implement greedy_action (see the reference in src/mlbook)')
