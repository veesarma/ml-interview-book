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

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(obs))  # (B, hidden)
        h = torch.relu(self.fc2(h))  # (B, hidden)
        return self.out(h)  # (B, A)


class DuelingQNetwork(nn.Module):
    """Dueling head (Wang et al., 2016): ``Q = V + (A - mean_a A)``.

    Subtracting the mean makes the decomposition identifiable (otherwise any
    constant could be moved between ``V`` and ``A``).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Linear(obs_dim, hidden)
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.trunk(obs))  # (B, hidden)
        v = self.value(h)  # (B, 1)
        adv = self.advantage(h)  # (B, A)
        return v + adv - adv.mean(dim=1, keepdim=True)  # (B, A)


class ReplayBuffer:
    """Ring buffer of transitions; uniform sampling breaks temporal correlation."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # (N, obs_dim)
        self.actions = np.zeros(capacity, dtype=np.int64)  # (N,)
        self.rewards = np.zeros(capacity, dtype=np.float32)  # (N,)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # (N, obs_dim)
        self.dones = np.zeros(capacity, dtype=np.float32)  # (N,)

    def add(self, obs: np.ndarray, a: int, r: float, next_obs: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, a, r
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, self.size, size=batch_size)  # (B,) uniform indices
        return (
            torch.from_numpy(self.obs[idx]),  # (B, obs_dim)
            torch.from_numpy(self.actions[idx]),  # (B,)
            torch.from_numpy(self.rewards[idx]),  # (B,)
            torch.from_numpy(self.next_obs[idx]),  # (B, obs_dim)
            torch.from_numpy(self.dones[idx]),  # (B,)
        )


def dqn_loss(
    q_net: nn.Module,
    target_net: nn.Module,
    batch: tuple[torch.Tensor, ...],
    gamma: float,
    double: bool = False,
) -> torch.Tensor:
    """Huber TD loss on a batch; returns a scalar tensor.

    Args:
        batch: ``(obs (B, obs_dim), actions (B,), rewards (B,), next_obs (B, obs_dim), dones (B,))``.
        double: use the online net to *select* ``a'`` and the target net to *evaluate* it.
    """
    obs, actions, rewards, next_obs, dones = batch
    q_all = q_net(obs)  # (B, A)
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,) Q(s, a) for the taken action
    with torch.no_grad():  # the target is a constant
        q_next_target = target_net(next_obs)  # (B, A)
        if double:
            a_star = q_net(next_obs).argmax(dim=1, keepdim=True)  # (B, 1) selected by online net
            q_next = q_next_target.gather(1, a_star).squeeze(1)  # (B,) evaluated by target net
        else:
            q_next = q_next_target.max(dim=1).values  # (B,)
        y = rewards + gamma * (1.0 - dones) * q_next  # (B,) bootstrapped target
    return nn.functional.smooth_l1_loss(q_sa, y)  # Huber, delta = 1


def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Linear anneal from ``eps_start`` to ``eps_end`` over ``decay_steps`` environment steps."""
    frac = min(1.0, step / max(1, decay_steps))
    return eps_start + frac * (eps_end - eps_start)


@dataclass
class DQNConfig:
    gamma: float = 0.98
    lr: float = 2e-3
    batch_size: int = 32
    buffer_size: int = 20_000
    warmup: int = 200
    target_update_every: int = 200
    train_every: int = 3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 1500
    double: bool = False
    hidden: int = 64


def train_dqn(env, n_episodes: int, cfg: DQNConfig, seed: int = 0, dueling: bool = False) -> tuple[nn.Module, list[float]]:
    """Run DQN on an episodic env with ``obs_dim`` / ``n_actions`` attributes.

    Returns the trained online network and the list of undiscounted episode returns.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net_cls = DuelingQNetwork if dueling else QNetwork
    q_net = net_cls(env.obs_dim, env.n_actions, cfg.hidden)
    target_net = copy.deepcopy(q_net)  # frozen copy, refreshed every target_update_every steps
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, env.obs_dim)
    returns: list[float] = []
    step = 0
    for _ in range(n_episodes):
        obs, done, ep_return = env.reset(rng), False, 0.0
        while not done:
            eps = epsilon_schedule(step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                with torch.no_grad():
                    q = q_net(torch.from_numpy(obs).unsqueeze(0))  # (1, A)
                a = int(q.argmax(dim=1).item())
            next_obs, r, done = env.step(a, rng)
            buffer.add(obs, a, r, next_obs, done)
            obs, ep_return, step = next_obs, ep_return + r, step + 1
            if buffer.size >= cfg.warmup and step % cfg.train_every == 0:
                loss = dqn_loss(q_net, target_net, buffer.sample(cfg.batch_size, rng), cfg.gamma, cfg.double)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step % cfg.target_update_every == 0:
                target_net.load_state_dict(q_net.state_dict())
        returns.append(ep_return)
    return q_net, returns


def greedy_action(q_net: nn.Module, obs: np.ndarray) -> int:
    """``argmax_a Q(obs, a)`` for a single observation (obs_dim,)."""
    with torch.no_grad():
        return int(q_net(torch.from_numpy(obs).unsqueeze(0)).argmax(dim=1).item())
