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

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))  # (B, hidden)
        return self.out(h)  # (B, A) unnormalised log-probabilities


def rewards_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = sum_{k>=0} gamma^k r_{t+k+1}``; (T,) -> (T,)."""
    G = np.zeros(len(rewards), dtype=np.float32)  # (T,)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def reinforce_loss(logp: torch.Tensor, returns: torch.Tensor, baseline: torch.Tensor | None = None) -> torch.Tensor:
    """``-(1/T) sum_t logp_t * (G_t - b_t)``; minimising it ascends ``J``.

    Args:
        logp: (T,) ``log pi(a_t|s_t)`` with gradients attached.
        returns: (T,) reward-to-go, no gradients.
        baseline: (T,) optional, no gradients (detached).
    """
    weights = returns if baseline is None else returns - baseline  # (T,)
    return -(logp * weights.detach()).mean()


def rollout(env, policy: PolicyNetwork, rng: np.random.Generator) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """One episode. Returns ``(obs (T, obs_dim), rewards (T,), actions (T,))``."""
    obs_list, rewards, actions = [], [], []
    obs, done = env.reset(rng), False
    while not done:
        with torch.no_grad():
            logits = policy(torch.from_numpy(obs).unsqueeze(0))  # (1, A)
        a = int(Categorical(logits=logits).sample().item())
        next_obs, r, done = env.step(a, rng)
        obs_list.append(obs)
        rewards.append(r)
        actions.append(a)
        obs = next_obs
    return torch.from_numpy(np.stack(obs_list)), np.asarray(rewards, dtype=np.float32), torch.tensor(actions)


def train_reinforce(
    env, n_episodes: int, gamma: float = 0.99, lr: float = 1e-2, seed: int = 0, use_baseline: bool = True
) -> tuple[PolicyNetwork, list[float]]:
    """Vanilla policy gradient, one update per episode, with a running-mean return baseline."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = PolicyNetwork(env.obs_dim, env.n_actions)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    returns_log: list[float] = []
    baseline_mean = 0.0
    for _ in range(n_episodes):
        obs, rewards, actions = rollout(env, policy, rng)  # (T, obs_dim), (T,), (T,)
        G = torch.from_numpy(rewards_to_go(rewards, gamma))  # (T,)
        logp = Categorical(logits=policy(obs)).log_prob(actions)  # (T,)
        baseline = torch.full_like(G, baseline_mean) if use_baseline else None
        loss = reinforce_loss(logp, G, baseline)
        opt.zero_grad()
        loss.backward()
        opt.step()
        baseline_mean = 0.9 * baseline_mean + 0.1 * float(G.mean())  # running mean of G_t, state-independent
        returns_log.append(float(rewards.sum()))
    return policy, returns_log
