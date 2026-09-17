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
    obs_l, act_l = [], []
    for _ in range(n_episodes):
        obs, done = env.reset(rng), False
        while not done:
            a = expert_fn(obs)
            obs_l.append(obs)
            act_l.append(a)
            obs, _, done = env.step(a, rng)
    return np.stack(obs_l), np.asarray(act_l, dtype=np.int64)


class TabularPolicy:
    """Histogram classifier over a grid of ``bin_width`` cells in observation space.

    A bin never seen in training returns the *class prior* (the most frequent expert action
    overall) -- what a classifier with uninformative features falls back to. This is the
    pessimistic "no generalisation" learner that makes the covariate-shift argument concrete.
    """

    def __init__(self, n_actions: int, bin_width: float, rng: np.random.Generator) -> None:
        self.n_actions, self.bin_width, self.rng = n_actions, bin_width, rng
        self.counts: dict[tuple[int, ...], np.ndarray] = {}
        self.prior = np.zeros(n_actions)  # (A,) global action counts

    def _bin(self, obs: np.ndarray) -> tuple[int, ...]:
        return tuple(int(np.floor(float(x) / self.bin_width)) for x in obs)

    def fit(self, obs: np.ndarray, actions: np.ndarray) -> None:
        """Accumulate action counts per bin from ``obs (N, obs_dim)`` and ``actions (N,)``."""
        for o, a in zip(obs, actions):
            b = self._bin(o)
            if b not in self.counts:
                self.counts[b] = np.zeros(self.n_actions)  # (A,)
            self.counts[b][a] += 1
            self.prior[a] += 1

    def act(self, obs: np.ndarray) -> int:
        b = self._bin(obs)
        if b not in self.counts:
            return int(np.argmax(self.prior))  # never seen: fall back to the marginal mode
        return int(np.argmax(self.counts[b]))

    def n_bins(self) -> int:
        return len(self.counts)


class MLPPolicy(nn.Module):
    """``obs (B, obs_dim) -> logits (B, A)``; trained with cross-entropy on expert actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.fc1(obs)))  # (B, A)

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(self(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())


def fit_bc_mlp(policy: MLPPolicy, obs: np.ndarray, actions: np.ndarray, n_epochs: int = 200, lr: float = 1e-2) -> float:
    """Full-batch cross-entropy training; returns the final loss.

    ``L = -(1/N) sum_i log softmax(f(s_i))[a_i]`` -- BC is a classification problem.
    """
    x = torch.from_numpy(obs.astype(np.float32))  # (N, obs_dim)
    y = torch.from_numpy(actions)  # (N,)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss = torch.zeros(())
    for _ in range(n_epochs):
        loss = nn.functional.cross_entropy(policy(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())
