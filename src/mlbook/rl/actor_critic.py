"""One-step advantage actor-critic (A2C-style, single environment) in PyTorch.

Separate actor ``pi_theta(a|s)`` and critic ``V_phi(s)``. Per episode::

    delta_t   = r_t + gamma (1 - d_t) V_phi(s_{t+1}) - V_phi(s_t)     # TD error = 1-step advantage
    L_actor   = -(1/T) sum_t log pi_theta(a_t|s_t) * stopgrad(delta_t)
    L_critic  =  (1/T) sum_t delta_t^2
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """``obs (B, obs_dim) -> logits (B, A)``."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.fc1(obs)))  # (B, A)


class Critic(nn.Module):
    """``obs (B, obs_dim) -> V (B,)``."""

    def __init__(self, obs_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.fc1(obs))).squeeze(-1)  # (B,)


def actor_critic_losses(
    actor: Actor,
    critic: Critic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(actor_loss, critic_loss)`` for a batch of ``T`` transitions.

    Shapes: obs/next_obs (T, obs_dim); actions (T,) int; rewards/dones (T,) float.
    """
    v = critic(obs)  # (T,)
    with torch.no_grad():
        v_next = critic(next_obs)  # (T,)
        target = rewards + gamma * (1.0 - dones) * v_next  # (T,) TD target
    delta = target - v  # (T,) TD error; gradient flows into v only
    logp = Categorical(logits=actor(obs)).log_prob(actions)  # (T,)
    actor_loss = -(logp * delta.detach()).mean()
    critic_loss = (delta**2).mean()
    return actor_loss, critic_loss


def train_actor_critic(
    env, n_episodes: int, gamma: float = 0.99, lr_actor: float = 3e-3, lr_critic: float = 1e-2, seed: int = 0
) -> tuple[Actor, Critic, list[float]]:
    """Episodic actor-critic: collect one episode, then take one actor and one critic step."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor, critic = Actor(env.obs_dim, env.n_actions), Critic(env.obs_dim)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    returns_log: list[float] = []
    for _ in range(n_episodes):
        obs_l, act_l, rew_l, next_l, done_l = [], [], [], [], []
        obs, done = env.reset(rng), False
        while not done:
            with torch.no_grad():
                a = int(Categorical(logits=actor(torch.from_numpy(obs).unsqueeze(0))).sample().item())
            next_obs, r, done = env.step(a, rng)
            obs_l.append(obs), act_l.append(a), rew_l.append(r), next_l.append(next_obs), done_l.append(float(done))
            obs = next_obs
        batch = (
            torch.from_numpy(np.stack(obs_l)),  # (T, obs_dim)
            torch.tensor(act_l),  # (T,)
            torch.tensor(rew_l, dtype=torch.float32),  # (T,)
            torch.from_numpy(np.stack(next_l)),  # (T, obs_dim)
            torch.tensor(done_l),  # (T,)
        )
        actor_loss, critic_loss = actor_critic_losses(actor, critic, *batch, gamma)
        opt_a.zero_grad(), opt_c.zero_grad()
        (actor_loss + critic_loss).backward()
        opt_a.step(), opt_c.step()
        returns_log.append(float(sum(rew_l)))
    return actor, critic, returns_log
