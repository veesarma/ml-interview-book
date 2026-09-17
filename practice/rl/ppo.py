# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/ppo.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ppo -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/ppo --force

"""Proximal Policy Optimisation with the clipped surrogate objective (PyTorch).

    r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t) = exp(logp_new - logp_old)
    L_clip     = E_t[ min( r_t A_t,  clip(r_t, 1-eps, 1+eps) A_t ) ]
    L          = -L_clip + c_v * (V_phi(s_t) - R_t)^2 - c_ent * H[pi_theta(.|s_t)]

Rollouts are collected with ``pi_old`` into a :class:`RolloutBuffer`; advantages
come from :func:`mlbook.rl.gae.compute_gae`; the same batch is reused for
``n_epochs`` of minibatch SGD, which is safe *because* the clip bounds how far
``pi_theta`` may move from ``pi_old``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from mlbook.rl.actor_critic import Actor, Critic
from mlbook.rl.gae import compute_gae
from mlbook.rl.reinforce import sample_action

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 0.003
    n_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 512
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalise_adv: bool = True
    target_kl: float | None = 0.03

class RolloutBuffer:
    """Fixed-length on-policy storage for one PPO iteration (``N = rollout_steps``)."""

    def __init__(self, n_steps: int, obs_dim: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def add(self, obs, a, logp, r, v, done) -> None:
        raise NotImplementedError('TODO: implement add (see the reference in src/mlbook)')

def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip_eps: float) -> torch.Tensor:
    """``-E[min(r A, clip(r, 1-eps, 1+eps) A)]`` with ``r = exp(logp_new - logp_old)``; all inputs (B,)."""
    raise NotImplementedError('TODO: implement ppo_clip_loss (see the reference in src/mlbook)')

def ppo_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Squared error between ``V_phi(s_t)`` and the GAE return target; inputs (B,)."""
    raise NotImplementedError('TODO: implement ppo_value_loss (see the reference in src/mlbook)')

def collect_rollout(env, actor: Actor, critic: Critic, buf: RolloutBuffer, rng: np.random.Generator, state: dict) -> float:
    """Fill ``buf`` with ``N`` steps of ``pi_old``; ``state`` carries the env across calls.

    Returns ``V(s_N)`` for bootstrapping a truncated final episode.
    """
    raise NotImplementedError('TODO: implement collect_rollout (see the reference in src/mlbook)')

def ppo_update(actor: Actor, critic: Critic, opt: torch.optim.Optimizer, buf: RolloutBuffer, last_value: float, cfg: PPOConfig) -> dict:
    """Several epochs of minibatch clipped-surrogate updates on one rollout; returns diagnostics."""
    raise NotImplementedError('TODO: implement ppo_update (see the reference in src/mlbook)')

def train_ppo(env, n_iterations: int, cfg: PPOConfig, seed: int=0) -> tuple[Actor, Critic, list[float]]:
    """Alternate ``collect_rollout`` and ``ppo_update``; returns actor, critic, completed-episode returns."""
    raise NotImplementedError('TODO: implement train_ppo (see the reference in src/mlbook)')
