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
    lr: float = 3e-3
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
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)  # (N, obs_dim)
        self.actions = np.zeros(n_steps, dtype=np.int64)  # (N,)
        self.logp = np.zeros(n_steps, dtype=np.float32)  # (N,) log pi_old(a|s)
        self.rewards = np.zeros(n_steps, dtype=np.float32)  # (N,)
        self.values = np.zeros(n_steps, dtype=np.float32)  # (N,) V_old(s)
        self.dones = np.zeros(n_steps, dtype=np.float32)  # (N,)
        self.ptr = 0

    def add(self, obs, a, logp, r, v, done) -> None:
        i = self.ptr
        self.obs[i], self.actions[i], self.logp[i] = obs, a, logp
        self.rewards[i], self.values[i], self.dones[i] = r, v, float(done)
        self.ptr += 1


def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip_eps: float) -> torch.Tensor:
    """``-E[min(r A, clip(r, 1-eps, 1+eps) A)]`` with ``r = exp(logp_new - logp_old)``; all inputs (B,)."""
    ratio = torch.exp(logp_new - logp_old)  # (B,) importance ratio, = 1 at the first epoch
    unclipped = ratio * adv  # (B,)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv  # (B,)
    return -torch.min(unclipped, clipped).mean()


def ppo_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Squared error between ``V_phi(s_t)`` and the GAE return target; inputs (B,)."""
    return ((values - returns) ** 2).mean()


def collect_rollout(env, actor: Actor, critic: Critic, buf: RolloutBuffer, rng: np.random.Generator, state: dict) -> float:
    """Fill ``buf`` with ``N`` steps of ``pi_old``; ``state`` carries the env across calls.

    Returns ``V(s_N)`` for bootstrapping a truncated final episode.
    """
    obs = state["obs"]
    for _ in range(buf.obs.shape[0]):
        a, logp = sample_action(actor, obs, rng)  # a ~ pi_old, logp = log pi_old(a|s)
        with torch.no_grad():
            v = float(critic(torch.from_numpy(obs).unsqueeze(0)).item())  # V_old(s)
        next_obs, r, done = env.step(a, rng)
        buf.add(obs, a, logp, r, v, done)
        state["ep_return"] += r
        if done:
            state["returns"].append(state["ep_return"])
            state["ep_return"] = 0.0
            next_obs = env.reset(rng)
        obs = next_obs
    state["obs"] = obs
    with torch.no_grad():
        return float(critic(torch.from_numpy(obs).unsqueeze(0)).item())


def ppo_update(
    actor: Actor, critic: Critic, opt: torch.optim.Optimizer, buf: RolloutBuffer, last_value: float, cfg: PPOConfig
) -> dict:
    """Several epochs of minibatch clipped-surrogate updates on one rollout; returns diagnostics."""
    adv_np, ret_np = compute_gae(buf.rewards, buf.values, buf.dones, last_value, cfg.gamma, cfg.lam)
    obs = torch.from_numpy(buf.obs)  # (N, obs_dim)
    actions = torch.from_numpy(buf.actions)  # (N,)
    logp_old = torch.from_numpy(buf.logp)  # (N,)
    adv = torch.from_numpy(adv_np).float()  # (N,)
    returns = torch.from_numpy(ret_np).float()  # (N,)
    if cfg.normalise_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    N = obs.shape[0]
    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "epochs_run": 0}
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(N)  # (N,) fresh shuffle every epoch
        for start in range(0, N, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]  # (B,)
            dist = Categorical(logits=actor(obs[idx]))
            logp_new = dist.log_prob(actions[idx])  # (B,)
            policy_loss = ppo_clip_loss(logp_new, logp_old[idx], adv[idx], cfg.clip_eps)
            value_loss = ppo_value_loss(critic(obs[idx]), returns[idx])
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                log_ratio = logp_new - logp_old[idx]  # (B,)
                stats["approx_kl"] = float((torch.exp(log_ratio) - 1.0 - log_ratio).mean())  # k3 estimator
                stats["clip_frac"] = float(((torch.exp(log_ratio) - 1.0).abs() > cfg.clip_eps).float().mean())
        stats["epochs_run"] = epoch + 1
        if cfg.target_kl is not None and stats["approx_kl"] > cfg.target_kl:
            break  # KL early stopping: pi_theta has moved far enough from pi_old
    return stats


def train_ppo(env, n_iterations: int, cfg: PPOConfig, seed: int = 0) -> tuple[Actor, Critic, list[float]]:
    """Alternate ``collect_rollout`` and ``ppo_update``; returns actor, critic, completed-episode returns."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor, critic = Actor(env.obs_dim, env.n_actions), Critic(env.obs_dim)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)
    state = {"obs": env.reset(rng), "ep_return": 0.0, "returns": []}
    for _ in range(n_iterations):
        buf = RolloutBuffer(cfg.rollout_steps, env.obs_dim)
        last_value = collect_rollout(env, actor, critic, buf, rng, state)
        ppo_update(actor, critic, opt, buf, last_value, cfg)
    return actor, critic, state["returns"]
