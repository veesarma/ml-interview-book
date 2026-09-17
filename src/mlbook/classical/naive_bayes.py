"""Naive Bayes: multinomial (counts / bag-of-words) with Laplace smoothing, and
Gaussian (continuous features).

Decision rule: ``ŷ = argmax_k [ log π_k + Σ_j log p(x_j | y = k) ]`` — the
"naive" part is the conditional independence of features given the class,
which turns a ``d``-dimensional density into ``d`` one-dimensional ones.
"""

from __future__ import annotations

import numpy as np


class MultinomialNB:
    """``p(x | y=k) ∝ Π_j θ_kj^{x_j}`` with ``θ_kj = (N_kj + α) / (N_k + α d)``.

    ``fit(X (N, d) non-negative counts, y (N,) ints)``; ``predict(X) -> (N,)``.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.log_prior: np.ndarray | None = None  # (K,)
        self.log_theta: np.ndarray | None = None  # (K, d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        K = int(y.max()) + 1
        d = X.shape[1]
        counts = np.zeros((K, d))  # (K, d)  N_kj = total count of feature j in class k
        class_n = np.zeros(K)  # (K,)  number of documents per class
        for k in range(K):
            counts[k] = X[y == k].sum(axis=0)  # (d,)
            class_n[k] = (y == k).sum()
        self.log_prior = np.log(class_n / len(y))  # (K,)
        theta = (counts + self.alpha) / (counts.sum(axis=1, keepdims=True) + self.alpha * d)  # (K, d)
        self.log_theta = np.log(theta)  # (K, d)
        return self

    def log_joint(self, X: np.ndarray) -> np.ndarray:
        """``log π_k + Σ_j x_j log θ_kj``. ``X``: (N, d) -> (N, K)."""
        return X @ self.log_theta.T + self.log_prior[None, :]  # (N, K)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.log_joint(X).argmax(axis=1)  # (N,)


class GaussianNB:
    """Per-class, per-feature Gaussians: ``p(x_j | y=k) = N(μ_kj, σ_kj²)``.

    ``fit(X (N, d) floats, y (N,) ints)``; ``predict(X) -> (N,)``.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.log_prior: np.ndarray | None = None  # (K,)
        self.mu: np.ndarray | None = None  # (K, d)
        self.var: np.ndarray | None = None  # (K, d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        K = int(y.max()) + 1
        d = X.shape[1]
        self.mu = np.zeros((K, d))  # (K, d)
        self.var = np.zeros((K, d))  # (K, d)
        prior = np.zeros(K)  # (K,)
        for k in range(K):
            Xk = X[y == k]  # (N_k, d)
            self.mu[k] = Xk.mean(axis=0)
            self.var[k] = Xk.var(axis=0) + self.var_smoothing * X.var(axis=0).max()
            prior[k] = len(Xk) / len(y)
        self.log_prior = np.log(prior)
        return self

    def log_joint(self, X: np.ndarray) -> np.ndarray:
        """``log π_k - ½ Σ_j [ log(2π σ_kj²) + (x_j - μ_kj)² / σ_kj² ]``. ``X``: (N, d) -> (N, K)."""
        diff = X[:, None, :] - self.mu[None, :, :]  # (N, K, d)
        quad = (diff * diff / self.var[None, :, :]).sum(axis=2)  # (N, K)
        log_norm = np.log(2 * np.pi * self.var).sum(axis=1)  # (K,)
        return self.log_prior[None, :] - 0.5 * (log_norm[None, :] + quad)  # (N, K)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.log_joint(X).argmax(axis=1)  # (N,)
