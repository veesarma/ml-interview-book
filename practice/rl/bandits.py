# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/bandits.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k bandits -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/bandits --force

"""Multi-armed and contextual bandits (pure NumPy).

A ``K``-armed Bernoulli bandit: pulling arm ``k`` returns ``r ~ Bernoulli(p_k)``.
Regret after ``T`` pulls is ``T * max_k p_k - sum_t p_{a_t}`` (expected reward
you gave up relative to the best arm). Each solver exposes ``select() -> k`` and
``update(k, r)``; :func:`run_bandit` records the per-step regret so that the
curves in the chapter figure can be reproduced.
"""
from __future__ import annotations
import numpy as np

class BernoulliBandit:
    """``K`` arms with success probabilities ``probs`` (K,)."""

    def __init__(self, probs: np.ndarray, rng: np.random.Generator) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @property
    def n_arms(self) -> int:
        raise NotImplementedError('TODO: implement n_arms (see the reference in src/mlbook)')

    def pull(self, k: int) -> float:
        raise NotImplementedError('TODO: implement pull (see the reference in src/mlbook)')

class EpsilonGreedy:
    """Pull argmax of empirical means w.p. ``1 - eps``, a random arm w.p. ``eps``."""

    def __init__(self, n_arms: int, eps: float, rng: np.random.Generator) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def select(self) -> int:
        raise NotImplementedError('TODO: implement select (see the reference in src/mlbook)')

    def update(self, k: int, r: float) -> None:
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

class UCB1:
    """Pull ``argmax_k  mean_k + c * sqrt( ln t / n_k )``.

    The bonus is the Hoeffding confidence radius: for rewards in [0, 1] and
    ``n_k`` samples, ``P(mean_k - p_k >= u) <= exp(-2 n_k u^2)``; setting this to
    ``t^-4`` and solving for ``u`` gives ``u = sqrt(2 ln t / n_k)``. ``c = sqrt(2)``
    reproduces the classic UCB1 of Auer, Cesa-Bianchi & Fischer (2002).
    """

    def __init__(self, n_arms: int, c: float=float(np.sqrt(2.0))) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def select(self) -> int:
        raise NotImplementedError('TODO: implement select (see the reference in src/mlbook)')

    def update(self, k: int, r: float) -> None:
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

class ThompsonBeta:
    """Bernoulli Thompson sampling with a ``Beta(alpha_k, beta_k)`` posterior per arm.

    Prior ``Beta(1, 1)`` (uniform). After a success ``alpha += 1``, after a failure
    ``beta += 1``; the posterior mean is ``alpha / (alpha + beta)``. Each step
    samples one plausible ``p_k`` per arm and pulls the argmax -- probability
    matching: arm ``k`` is chosen with the posterior probability that it is best.
    """

    def __init__(self, n_arms: int, rng: np.random.Generator) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def select(self) -> int:
        raise NotImplementedError('TODO: implement select (see the reference in src/mlbook)')

    def update(self, k: int, r: float) -> None:
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

def run_bandit(bandit: BernoulliBandit, solver, horizon: int) -> np.ndarray:
    """Interact for ``horizon`` steps; return the cumulative pseudo-regret ``(T,)``."""
    raise NotImplementedError('TODO: implement run_bandit (see the reference in src/mlbook)')

class LinUCB:
    """Disjoint LinUCB (Li, Chu, Langford & Schapire, WWW 2010) for contextual bandits.

    Arm ``k`` has its own ridge regression ``theta_k = A_k^-1 b_k`` of reward on the
    context ``x`` (d,). Score ``x . theta_k + alpha * sqrt(x^T A_k^-1 x)``: the
    second term is the width of the confidence ellipsoid along ``x``.
    """

    def __init__(self, n_arms: int, d: int, alpha: float=1.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def select(self, x: np.ndarray) -> int:
        raise NotImplementedError('TODO: implement select (see the reference in src/mlbook)')

    def update(self, k: int, x: np.ndarray, r: float) -> None:
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

def run_linear_contextual(theta_true: np.ndarray, solver: LinUCB, horizon: int, rng: np.random.Generator, noise: float=0.1) -> np.ndarray:
    """Synthetic contextual bandit: reward ``x . theta_true[k] + noise``; returns cumulative regret (T,)."""
    raise NotImplementedError('TODO: implement run_linear_contextual (see the reference in src/mlbook)')
