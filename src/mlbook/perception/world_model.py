"""A tiny latent world model and planning in its latent space (PyTorch).

    z_t = enc(x_t),   ẑ_{t+1} = z_t + f(z_t, a_t),   x̂_t = dec(z_t)

trained with reconstruction ``‖dec(z_t) − x_t‖²`` and latent consistency
``‖ẑ_{t+1} − sg(enc(x_{t+1}))‖²`` (the deterministic core of an RSSM without the stochastic
state).  With a learned transition you can plan without touching the real system: sample
action sequences, roll them forward in latent space, score the imagined trajectory, and
keep refining (random shooting / cross-entropy method).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn


class PointMassDynamics:
    """Toy environment: 2-D point mass, state ``[x, y, v_x, v_y]``, action = acceleration.

    ``v ← v + a·dt``, ``p ← p + v·dt``; observation = state plus optional Gaussian noise.
    """

    def __init__(self, dt: float = 0.1, obs_noise: float = 0.0):
        self.dt, self.obs_noise = dt, obs_noise

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """state (N, 4), action (N, 2) → next state (N, 4)."""
        v = state[:, 2:] + action * self.dt  # (N, 2)
        p = state[:, :2] + v * self.dt  # (N, 2)
        return np.concatenate([p, v], axis=1)  # (N, 4)

    def observe(self, state: np.ndarray) -> np.ndarray:
        return state + self.obs_noise * np.random.randn(*state.shape)  # (N, 4)

    def rollout(self, state0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """state0 (N, 4), actions (N, T, 2) → observations (N, T+1, 4)."""
        obs = [self.observe(state0)]
        s = state0
        for t in range(actions.shape[1]):
            s = self.step(s, actions[:, t])
            obs.append(self.observe(s))
        return np.stack(obs, axis=1)  # (N, T+1, 4)


class LatentWorldModel(nn.Module):
    """encoder → latent, action-conditioned residual transition, decoder."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.transition = nn.Sequential(nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)  # (N, latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # (N, obs)

    def next_latent(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``ẑ' = z + f([z; a])``: (N, latent), (N, action) → (N, latent)."""
        return z + self.transition(torch.cat([z, a], dim=-1))  # (N, latent)

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0 (N, latent), actions (N, T, action) → imagined latents (N, T+1, latent)."""
        zs = [z0]
        z = z0
        for t in range(actions.shape[1]):
            z = self.next_latent(z, actions[:, t])  # (N, latent)
            zs.append(z)
        return torch.stack(zs, dim=1)  # (N, T+1, latent)


def world_model_loss(model: LatentWorldModel, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Reconstruction + latent consistency on a batch of trajectories.

    obs (N, T+1, obs_dim), actions (N, T, action_dim).  The consistency target is the
    encoder's own next latent with the gradient stopped, so the transition chases the
    encoder and the encoder is shaped by reconstruction only (avoids latent collapse).
    """
    n, t1, _ = obs.shape
    z = model.encode(obs)  # (N, T+1, latent)
    recon = ((model.decode(z) - obs) ** 2).mean()
    z_pred = model.next_latent(z[:, :-1].reshape(-1, z.shape[-1]), actions.reshape(-1, actions.shape[-1]))  # (N·T, latent)
    z_target = z[:, 1:].reshape(-1, z.shape[-1]).detach()  # (N·T, latent)
    consistency = ((z_pred - z_target) ** 2).mean()
    return recon + consistency


def random_shooting_plan(model: LatentWorldModel, z0: torch.Tensor, cost_fn: Callable[[torch.Tensor], torch.Tensor],
                         horizon: int, action_dim: int, n_samples: int = 256, action_scale: float = 1.0) -> torch.Tensor:
    """Sample ``n_samples`` action sequences, imagine them, return the cheapest (T, action_dim)."""
    with torch.no_grad():
        actions = (torch.rand(n_samples, horizon, action_dim) * 2.0 - 1.0) * action_scale  # (S, T, a)
        zs = model.rollout(z0.expand(n_samples, -1), actions)  # (S, T+1, latent)
        cost = cost_fn(model.decode(zs))  # (S,)
        return actions[cost.argmin()]  # (T, a)


def cem_plan(model: LatentWorldModel, z0: torch.Tensor, cost_fn: Callable[[torch.Tensor], torch.Tensor], horizon: int,
             action_dim: int, n_samples: int = 256, n_elite: int = 32, n_iters: int = 5, action_scale: float = 1.0) -> torch.Tensor:
    """Cross-entropy method: fit a Gaussian over action sequences to the elite set, repeat.

    Returns the final mean (T, action_dim).  ``cost_fn`` maps decoded observations
    (S, T+1, obs_dim) → cost (S,).
    """
    mean = torch.zeros(horizon, action_dim)  # (T, a)
    std = torch.full((horizon, action_dim), action_scale)  # (T, a)
    with torch.no_grad():
        for _ in range(n_iters):
            actions = (mean + std * torch.randn(n_samples, horizon, action_dim)).clamp(-action_scale, action_scale)  # (S, T, a)
            zs = model.rollout(z0.expand(n_samples, -1), actions)  # (S, T+1, latent)
            cost = cost_fn(model.decode(zs))  # (S,)
            elite = actions[cost.topk(n_elite, largest=False).indices]  # (E, T, a)
            mean, std = elite.mean(0), elite.std(0) + 1e-3  # (T, a) each
    return mean
