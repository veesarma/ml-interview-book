# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/dagger.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k dagger -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/dagger --force

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

def rollout_and_relabel(env, learner, expert_fn, n_episodes: int, rng: np.random.Generator, beta: float=0.0):
    """Act with ``learner`` (expert w.p. ``beta``); return ``(obs (N, obs_dim), expert_actions (N,), mean_return)``."""
    raise NotImplementedError('TODO: implement rollout_and_relabel (see the reference in src/mlbook)')

def dagger(env, expert_fn, make_learner, n_iterations: int, episodes_per_iter: int, rng: np.random.Generator, beta0: float=0.5) -> tuple[object, list[float]]:
    """Run DAgger; ``make_learner()`` returns a fresh learner with ``fit`` / ``act``.

    ``beta_i = beta0 ** i`` mixes in the expert early on (the paper's schedule).
    Returns the final learner and the per-iteration mean return of the *learner-driven* rollouts.
    """
    raise NotImplementedError('TODO: implement dagger (see the reference in src/mlbook)')

def make_tabular_learner(n_actions: int, bin_width: float, rng: np.random.Generator):
    """Factory used by tests and the chapter figure."""
    raise NotImplementedError('TODO: implement make_tabular_learner (see the reference in src/mlbook)')
