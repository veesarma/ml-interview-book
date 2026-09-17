# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/world_model.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k world_model -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/world_model --force

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

    def __init__(self, dt: float=0.1, obs_noise: float=0.0):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """state (N, 4), action (N, 2) → next state (N, 4)."""
        raise NotImplementedError('TODO: implement step (see the reference in src/mlbook)')

    def observe(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement observe (see the reference in src/mlbook)')

    def rollout(self, state0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """state0 (N, 4), actions (N, T, 2) → observations (N, T+1, 4)."""
        raise NotImplementedError('TODO: implement rollout (see the reference in src/mlbook)')

class LatentWorldModel(nn.Module):
    """encoder → latent, action-conditioned residual transition, decoder."""

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int=16, hidden: int=64):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def next_latent(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``ẑ' = z + f([z; a])``: (N, latent), (N, action) → (N, latent)."""
        raise NotImplementedError('TODO: implement next_latent (see the reference in src/mlbook)')

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0 (N, latent), actions (N, T, action) → imagined latents (N, T+1, latent)."""
        raise NotImplementedError('TODO: implement rollout (see the reference in src/mlbook)')

def world_model_loss(model: LatentWorldModel, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Reconstruction + latent consistency on a batch of trajectories.

    obs (N, T+1, obs_dim), actions (N, T, action_dim).  The consistency target is the
    encoder's own next latent with the gradient stopped, so the transition chases the
    encoder and the encoder is shaped by reconstruction only (avoids latent collapse).
    """
    raise NotImplementedError('TODO: implement world_model_loss (see the reference in src/mlbook)')

def random_shooting_plan(model: LatentWorldModel, z0: torch.Tensor, cost_fn: Callable[[torch.Tensor], torch.Tensor], horizon: int, action_dim: int, n_samples: int=256, action_scale: float=1.0) -> torch.Tensor:
    """Sample ``n_samples`` action sequences, imagine them, return the cheapest (T, action_dim)."""
    raise NotImplementedError('TODO: implement random_shooting_plan (see the reference in src/mlbook)')

def cem_plan(model: LatentWorldModel, z0: torch.Tensor, cost_fn: Callable[[torch.Tensor], torch.Tensor], horizon: int, action_dim: int, n_samples: int=256, n_elite: int=32, n_iters: int=5, action_scale: float=1.0) -> torch.Tensor:
    """Cross-entropy method: fit a Gaussian over action sequences to the elite set, repeat.

    Returns the final mean (T, action_dim).  ``cost_fn`` maps decoded observations
    (S, T+1, obs_dim) → cost (S,).
    """
    raise NotImplementedError('TODO: implement cem_plan (see the reference in src/mlbook)')
