# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/actor_critic.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k actor_critic -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/actor_critic --force

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
from mlbook.rl.reinforce import sample_action

class Actor(nn.Module):
    """``obs (B, obs_dim) -> logits (B, A)``."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class Critic(nn.Module):
    """``obs (B, obs_dim) -> V (B,)``."""

    def __init__(self, obs_dim: int, hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def actor_critic_losses(actor: Actor, critic: Critic, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor, dones: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(actor_loss, critic_loss)`` for a batch of ``T`` transitions.

    Shapes: obs/next_obs (T, obs_dim); actions (T,) int; rewards/dones (T,) float.
    """
    raise NotImplementedError('TODO: implement actor_critic_losses (see the reference in src/mlbook)')

def train_actor_critic(env, n_episodes: int, gamma: float=0.99, lr_actor: float=0.003, lr_critic: float=0.01, seed: int=0) -> tuple[Actor, Critic, list[float]]:
    """Episodic actor-critic: collect one episode, then take one actor and one critic step."""
    raise NotImplementedError('TODO: implement train_actor_critic (see the reference in src/mlbook)')
