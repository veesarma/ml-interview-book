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
        self.probs = np.asarray(probs, dtype=float)  # (K,)
        self.rng = rng
        self.best = float(self.probs.max())

    @property
    def n_arms(self) -> int:
        return self.probs.shape[0]

    def pull(self, k: int) -> float:
        return float(self.rng.random() < self.probs[k])


class EpsilonGreedy:
    """Pull argmax of empirical means w.p. ``1 - eps``, a random arm w.p. ``eps``."""

    def __init__(self, n_arms: int, eps: float, rng: np.random.Generator) -> None:
        self.eps, self.rng = eps, rng
        self.counts = np.zeros(n_arms)  # (K,) pulls per arm
        self.sums = np.zeros(n_arms)  # (K,) reward sums per arm

    def select(self) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.counts.shape[0]))
        means = self.sums / np.maximum(self.counts, 1)  # (K,) unpulled arms score 0
        return int(np.argmax(means))

    def update(self, k: int, r: float) -> None:
        self.counts[k] += 1
        self.sums[k] += r


class UCB1:
    """Pull ``argmax_k  mean_k + c * sqrt( ln t / n_k )``.

    The bonus is the Hoeffding confidence radius: for rewards in [0, 1] and
    ``n_k`` samples, ``P(mean_k - p_k >= u) <= exp(-2 n_k u^2)``; setting this to
    ``t^-4`` and solving for ``u`` gives ``u = sqrt(2 ln t / n_k)``. ``c = sqrt(2)``
    reproduces the classic UCB1 of Auer, Cesa-Bianchi & Fischer (2002).
    """

    def __init__(self, n_arms: int, c: float = float(np.sqrt(2.0))) -> None:
        self.c = c
        self.t = 0
        self.counts = np.zeros(n_arms)  # (K,)
        self.sums = np.zeros(n_arms)  # (K,)

    def select(self) -> int:
        self.t += 1
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))  # pull every arm once first
        means = self.sums / self.counts  # (K,)
        bonus = self.c * np.sqrt(np.log(self.t) / self.counts)  # (K,) confidence radius
        return int(np.argmax(means + bonus))

    def update(self, k: int, r: float) -> None:
        self.counts[k] += 1
        self.sums[k] += r


class ThompsonBeta:
    """Bernoulli Thompson sampling with a ``Beta(alpha_k, beta_k)`` posterior per arm.

    Prior ``Beta(1, 1)`` (uniform). After a success ``alpha += 1``, after a failure
    ``beta += 1``; the posterior mean is ``alpha / (alpha + beta)``. Each step
    samples one plausible ``p_k`` per arm and pulls the argmax -- probability
    matching: arm ``k`` is chosen with the posterior probability that it is best.
    """

    def __init__(self, n_arms: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.alpha = np.ones(n_arms)  # (K,) successes + 1
        self.beta = np.ones(n_arms)  # (K,) failures + 1

    def select(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)  # (K,) one draw per posterior
        return int(np.argmax(samples))

    def update(self, k: int, r: float) -> None:
        self.alpha[k] += r
        self.beta[k] += 1.0 - r


def run_bandit(bandit: BernoulliBandit, solver, horizon: int) -> np.ndarray:
    """Interact for ``horizon`` steps; return the cumulative pseudo-regret ``(T,)``."""
    regret = np.zeros(horizon)  # (T,)
    total = 0.0
    for t in range(horizon):
        k = solver.select()
        r = bandit.pull(k)
        solver.update(k, r)
        total += bandit.best - bandit.probs[k]
        regret[t] = total
    return regret


class LinUCB:
    """Disjoint LinUCB (Li, Chu, Langford & Schapire, WWW 2010) for contextual bandits.

    Arm ``k`` has its own ridge regression ``theta_k = A_k^-1 b_k`` of reward on the
    context ``x`` (d,). Score ``x . theta_k + alpha * sqrt(x^T A_k^-1 x)``: the
    second term is the width of the confidence ellipsoid along ``x``.
    """

    def __init__(self, n_arms: int, d: int, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.A = np.stack([np.eye(d) for _ in range(n_arms)])  # (K, d, d) ridge Gram matrices
        self.b = np.zeros((n_arms, d))  # (K, d) reward-weighted context sums

    def select(self, x: np.ndarray) -> int:
        A_inv = np.linalg.inv(self.A)  # (K, d, d)
        theta = np.einsum("kij,kj->ki", A_inv, self.b)  # (K, d): theta_k = A_k^-1 b_k
        mean = theta @ x  # (K,) predicted reward per arm
        width = np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))  # (K,) sqrt(x^T A_k^-1 x)
        return int(np.argmax(mean + self.alpha * width))

    def update(self, k: int, x: np.ndarray, r: float) -> None:
        self.A[k] += np.outer(x, x)  # (d, d)
        self.b[k] += r * x  # (d,)


def run_linear_contextual(
    theta_true: np.ndarray, solver: LinUCB, horizon: int, rng: np.random.Generator, noise: float = 0.1
) -> np.ndarray:
    """Synthetic contextual bandit: reward ``x . theta_true[k] + noise``; returns cumulative regret (T,)."""
    K, d = theta_true.shape
    regret = np.zeros(horizon)  # (T,)
    total = 0.0
    for t in range(horizon):
        x = rng.normal(size=d)  # (d,) context
        x = x / np.linalg.norm(x)
        expected = theta_true @ x  # (K,) true expected reward per arm
        k = solver.select(x)
        r = float(expected[k] + noise * rng.normal())
        solver.update(k, x, r)
        total += float(expected.max() - expected[k])
        regret[t] = total
    return regret
