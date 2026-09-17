"""Gaussian discriminant analysis: LDA (shared covariance) and QDA (per-class).

``p(x | y=k) = N(μ_k, Σ_k)``, ``p(y=k) = π_k``. With a shared ``Σ`` the
posterior ``p(y=1 | x)`` is exactly a logistic function of a *linear* score,
which :func:`lda_as_logistic` returns explicitly.
"""

from __future__ import annotations

import numpy as np


class GDA:
    """``shared_cov=True`` gives LDA (linear boundaries); ``False`` gives QDA.

    ``fit(X (N, d), y (N,) ints)``; ``predict(X) -> (N,)``; ``log_posterior(X) -> (N, K)``.
    """

    def __init__(self, shared_cov: bool = True, reg: float = 1e-6) -> None:
        self.shared_cov = shared_cov
        self.reg = reg
        self.pi: np.ndarray | None = None  # (K,)
        self.mu: np.ndarray | None = None  # (K, d)
        self.Sigma: np.ndarray | None = None  # (K, d, d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GDA":
        N, d = X.shape
        K = int(y.max()) + 1
        self.pi = np.zeros(K)  # (K,)
        self.mu = np.zeros((K, d))  # (K, d)
        self.Sigma = np.zeros((K, d, d))  # (K, d, d)
        for k in range(K):
            Xk = X[y == k]  # (N_k, d)
            self.pi[k] = len(Xk) / N
            self.mu[k] = Xk.mean(axis=0)
            centred = Xk - self.mu[k]  # (N_k, d)
            self.Sigma[k] = centred.T @ centred / len(Xk) + self.reg * np.eye(d)  # (d, d)
        if self.shared_cov:
            pooled = sum(self.pi[k] * self.Sigma[k] for k in range(K))  # (d, d) weighted average
            self.Sigma = np.repeat(pooled[None], K, axis=0)  # (K, d, d)
        return self

    def log_posterior(self, X: np.ndarray) -> np.ndarray:
        """Unnormalised ``log π_k + log N(x; μ_k, Σ_k)``. ``X``: (N, d) -> (N, K)."""
        K, d = self.mu.shape
        out = np.empty((X.shape[0], K))  # (N, K)
        for k in range(K):
            diff = X - self.mu[k]  # (N, d)
            sign, logdet = np.linalg.slogdet(self.Sigma[k])
            sol = np.linalg.solve(self.Sigma[k], diff.T).T  # (N, d)   Σ^{-1} (x - μ)
            maha = (diff * sol).sum(axis=1)  # (N,)
            out[:, k] = np.log(self.pi[k]) - 0.5 * (logdet + maha + d * np.log(2 * np.pi))
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.log_posterior(X).argmax(axis=1)  # (N,)


def lda_as_logistic(model: GDA) -> tuple[np.ndarray, float]:
    """Return ``(w, b)`` with ``p(y=1 | x) = σ(w·x + b)`` for a two-class LDA.

    ``w = Σ^{-1} (μ_1 - μ_0)``,
    ``b = -½ μ_1^T Σ^{-1} μ_1 + ½ μ_0^T Σ^{-1} μ_0 + log(π_1 / π_0)``.
    """
    Sigma = model.Sigma[0]  # (d, d)
    mu0, mu1 = model.mu[0], model.mu[1]  # (d,), (d,)
    w = np.linalg.solve(Sigma, mu1 - mu0)  # (d,)
    b = -0.5 * mu1 @ np.linalg.solve(Sigma, mu1) + 0.5 * mu0 @ np.linalg.solve(Sigma, mu0)
    b += np.log(model.pi[1] / model.pi[0])
    return w, float(b)
