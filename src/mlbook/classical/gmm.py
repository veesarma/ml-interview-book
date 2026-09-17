"""Gaussian mixture model fitted with EM.

E-step: responsibilities ``r_ik = π_k N(x_i; μ_k, Σ_k) / Σ_j π_j N(x_i; μ_j, Σ_j)``.
M-step: ``N_k = Σ_i r_ik``, ``π_k = N_k / N``, ``μ_k = (1/N_k) Σ_i r_ik x_i``,
``Σ_k = (1/N_k) Σ_i r_ik (x_i - μ_k)(x_i - μ_k)^T``.
The log-likelihood is non-decreasing across iterations (tracked in ``history_``).
"""

from __future__ import annotations

import numpy as np

from .kmeans import kmeans_plusplus_init


def log_gaussian(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """``log N(x_i; μ, Σ)`` for every row. ``X``: (N, d), ``mu``: (d,), ``Sigma``: (d, d) -> (N,)."""
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)  # (d, d)  Σ = L L^T
    diff = X - mu  # (N, d)
    z = np.linalg.solve(L, diff.T).T  # (N, d)   L^{-1}(x - μ), so ||z||² = (x-μ)^T Σ^{-1} (x-μ)
    maha = (z * z).sum(axis=1)  # (N,)
    logdet = 2.0 * np.log(np.diag(L)).sum()  # scalar  log|Σ|
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)  # (N,)


def logsumexp(A: np.ndarray, axis: int = 1) -> np.ndarray:
    """Stable ``log Σ exp`` along ``axis``."""
    m = A.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class GMM:
    """``fit(X (N, d))``; ``predict_proba(X) -> (N, k)`` responsibilities;
    ``score(X) -> float`` mean log-likelihood; ``predict(X) -> (N,)`` hard labels."""

    def __init__(self, k: int, n_iters: int = 100, reg: float = 1e-6, seed: int = 0, tol: float = 1e-8) -> None:
        self.k, self.n_iters, self.reg, self.tol = k, n_iters, reg, tol
        self.rng = np.random.default_rng(seed)
        self.pi: np.ndarray | None = None  # (k,)
        self.mu: np.ndarray | None = None  # (k, d)
        self.Sigma: np.ndarray | None = None  # (k, d, d)
        self.history_: list[float] = []

    def _log_joint(self, X: np.ndarray) -> np.ndarray:
        """``log π_k + log N(x_i; μ_k, Σ_k)``. ``X``: (N, d) -> (N, k)."""
        out = np.empty((X.shape[0], self.k))  # (N, k)
        for j in range(self.k):
            out[:, j] = np.log(self.pi[j]) + log_gaussian(X, self.mu[j], self.Sigma[j])
        return out

    def e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Responsibilities (N, k) and the total log-likelihood ``Σ_i log Σ_k π_k N(...)``."""
        lj = self._log_joint(X)  # (N, k)
        lse = logsumexp(lj, axis=1)  # (N,)   log p(x_i)
        R = np.exp(lj - lse[:, None])  # (N, k)  rows sum to 1
        return R, float(lse.sum())

    def m_step(self, X: np.ndarray, R: np.ndarray) -> None:
        """Closed-form updates given responsibilities ``R`` (N, k)."""
        N, d = X.shape
        Nk = R.sum(axis=0) + 1e-12  # (k,)
        self.pi = Nk / N  # (k,)
        self.mu = (R.T @ X) / Nk[:, None]  # (k, d)
        for j in range(self.k):
            diff = X - self.mu[j]  # (N, d)
            self.Sigma[j] = (R[:, j, None] * diff).T @ diff / Nk[j] + self.reg * np.eye(d)  # (d, d)

    def fit(self, X: np.ndarray) -> "GMM":
        N, d = X.shape
        self.mu = kmeans_plusplus_init(X, self.k, self.rng)  # (k, d)
        self.pi = np.full(self.k, 1.0 / self.k)  # (k,)
        self.Sigma = np.repeat((np.cov(X.T) + self.reg * np.eye(d))[None], self.k, axis=0)  # (k, d, d)
        self.history_ = []
        prev = -np.inf
        for _ in range(self.n_iters):
            R, ll = self.e_step(X)  # (N, k), scalar
            self.history_.append(ll)
            self.m_step(X, R)
            if ll - prev < self.tol:
                break
            prev = ll
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.e_step(X)[0]  # (N, k)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)  # (N,)

    def score(self, X: np.ndarray) -> float:
        """Mean per-example log-likelihood."""
        return self.e_step(X)[1] / X.shape[0]
