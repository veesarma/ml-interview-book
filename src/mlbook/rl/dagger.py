"""DAgger (Ross, Gordon & Bagnell, 2011): dataset aggregation with expert relabelling.

    D <- expert demonstrations
    repeat:
        pi_i <- fit(D)
        roll out pi_i (optionally mixed with the expert), record visited states
        label those states with the EXPERT's action; D <- D  U  {(s, expert(s))}

The learner is trained on its *own* state distribution, which is what plain
behavioural cloning never sees.
"""

from __future__ import annotations

import numpy as np

from mlbook.rl.behavioral_cloning import TabularPolicy, collect_expert_data


def rollout_and_relabel(env, learner, expert_fn, n_episodes: int, rng: np.random.Generator, beta: float = 0.0):
    """Act with ``learner`` (expert w.p. ``beta``); return ``(obs (N, obs_dim), expert_actions (N,), mean_return)``."""
    obs_l, act_l, returns = [], [], []
    for _ in range(n_episodes):
        obs, done, total = env.reset(rng), False, 0.0
        while not done:
            a_expert = expert_fn(obs)
            a = a_expert if rng.random() < beta else learner.act(obs)
            obs_l.append(obs)
            act_l.append(a_expert)  # label = what the expert WOULD do here
            obs, r, done = env.step(a, rng)
            total += r
        returns.append(total)
    return np.stack(obs_l), np.asarray(act_l, dtype=np.int64), float(np.mean(returns))


def dagger(
    env,
    expert_fn,
    make_learner,
    n_iterations: int,
    episodes_per_iter: int,
    rng: np.random.Generator,
    beta0: float = 0.5,
) -> tuple[object, list[float]]:
    """Run DAgger; ``make_learner()`` returns a fresh learner with ``fit`` / ``act``.

    ``beta_i = beta0 ** i`` mixes in the expert early on (the paper's schedule).
    Returns the final learner and the per-iteration mean return of the *learner-driven* rollouts.
    """
    obs, actions = collect_expert_data(env, expert_fn, episodes_per_iter, rng)  # (N, obs_dim), (N,)
    history: list[float] = []
    learner = make_learner()
    for i in range(n_iterations):
        learner = make_learner()
        learner.fit(obs, actions)
        beta = beta0**i
        new_obs, new_actions, mean_return = rollout_and_relabel(env, learner, expert_fn, episodes_per_iter, rng, beta)
        obs = np.concatenate([obs, new_obs], axis=0)  # (N + N_new, obs_dim)
        actions = np.concatenate([actions, new_actions], axis=0)  # (N + N_new,)
        history.append(mean_return)
    learner = make_learner()
    learner.fit(obs, actions)
    return learner, history


def make_tabular_learner(n_actions: int, bin_width: float, rng: np.random.Generator):
    """Factory used by tests and the chapter figure."""
    return lambda: TabularPolicy(n_actions, bin_width, rng)
