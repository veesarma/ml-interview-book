"""Kernels, kernel ridge regression and Nadaraya–Watson kernel smoothing.

A kernel ``κ(x, x')`` is an inner product in a feature space:
``κ(x, x') = φ(x)·φ(x')``. Mercer's condition (the Gram matrix ``K_ij = κ(x_i, x_j)``
is PSD for every finite set) is what makes that feature space exist.
"""

from __future__ import annotations

import numpy as np


def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``K_ij = a_i · b_j``. ``A``: (N, d), ``B``: (M, d) -> (N, M)."""
    return A @ B.T  # (N, M)


def polynomial_kernel(A: np.ndarray, B: np.ndarray, degree: int = 3, coef0: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """``K_ij = (γ a_i·b_j + c)^p``. Feature space = all monomials up to degree ``p``. (N, M)."""
    return (gamma * A @ B.T + coef0) ** degree  # (N, M)


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """``K_ij = exp(-γ ||a_i - b_j||²)``: infinite-dimensional feature space. (N, M)."""
    a2 = (A * A).sum(axis=1, keepdims=True)  # (N, 1)
    b2 = (B * B).sum(axis=1)[None, :]  # (1, M)
    sq = np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)  # (N, M)
    return np.exp(-gamma * sq)  # (N, M)


class KernelRidge:
    """Kernel ridge regression: ``α = (K + λ I)^{-1} y``, ``f(x) = Σ_i α_i κ(x_i, x)``.

    Derivation: minimise ``||Φw - y||² + λ||w||²`` with ``w = Φ^T α`` (representer
    theorem) and ``K = Φ Φ^T``. Training is ``O(N³)``; prediction ``O(N)`` per query.
    ``fit(X (N, d), y (N,))``; ``predict(Q (M, d)) -> (M,)``.
    """

    def __init__(self, kernel, lam: float = 1.0) -> None:
        self.kernel = kernel
        self.lam = lam
        self.X: np.ndarray | None = None
        self.alpha: np.ndarray | None = None  # (N,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidge":
        K = self.kernel(X, X)  # (N, N)
        self.alpha = np.linalg.solve(K + self.lam * np.eye(len(y)), y)  # (N,)
        self.X = X
        return self

    def predict(self, Q: np.ndarray) -> np.ndarray:
        return self.kernel(Q, self.X) @ self.alpha  # (M,)


def nadaraya_watson(Q: np.ndarray, X: np.ndarray, y: np.ndarray, kernel) -> np.ndarray:
    """Kernel smoother ``f(q) = Σ_i κ(q, x_i) y_i / Σ_i κ(q, x_i)``.

    This is exactly attention with ``q`` as query, ``x_i`` as keys, ``y_i`` as values
    and ``κ`` as the (unnormalised) score; softmax attention uses ``κ = exp(q·k/√d)``.
    ``Q``: (M, d), ``X``: (N, d), ``y``: (N, v) or (N,) -> (M, v) or (M,).
    """
    K = kernel(Q, X)  # (M, N)
    W = K / K.sum(axis=1, keepdims=True)  # (M, N) rows sum to 1 (the "attention weights")
    return W @ y  # (M, v)
