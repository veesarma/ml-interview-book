"""Support vector machines: the soft-margin dual solved with a simplified SMO,
and the primal hinge-loss formulation solved by (sub)gradient descent.

Dual (labels ``y ∈ {-1, +1}``):

    max_α  Σ_i α_i - ½ Σ_ij α_i α_j y_i y_j κ(x_i, x_j)
    s.t.   0 ≤ α_i ≤ C,   Σ_i α_i y_i = 0.

``f(x) = Σ_i α_i y_i κ(x_i, x) + b``; support vectors are the ``i`` with ``α_i > 0``.
"""

from __future__ import annotations

import numpy as np


class SVM:
    """Kernel soft-margin SVM trained with SMO-lite (Platt 1998, random second index).

    ``fit(X (N, d), y (N,) in {-1,+1})``; ``decision_function(Q (M, d)) -> (M,)``;
    ``predict(Q) -> (M,)`` in {-1,+1}.
    """

    def __init__(self, kernel, C: float = 1.0, max_passes: int = 10, tol: float = 1e-3, seed: int = 0) -> None:
        self.kernel = kernel
        self.C = C
        self.max_passes = max_passes
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.alpha: np.ndarray | None = None  # (N,)
        self.b = 0.0

    def _f(self, K: np.ndarray, i: int) -> float:
        """``f(x_i) = Σ_j α_j y_j K_ji + b``."""
        return float((self.alpha * self.y) @ K[:, i] + self.b)

    def _update_pair(self, K: np.ndarray, i: int, j: int) -> bool:
        """Optimise ``(α_i, α_j)`` analytically with all other α fixed. Returns True if changed."""
        y, a, C = self.y, self.alpha, self.C
        E_i = self._f(K, i) - y[i]
        E_j = self._f(K, j) - y[j]
        # Box constraints from Σ α y = 0 confine α_j to [L, H].
        if y[i] != y[j]:
            L, H = max(0.0, a[j] - a[i]), min(C, C + a[j] - a[i])
        else:
            L, H = max(0.0, a[i] + a[j] - C), min(C, a[i] + a[j])
        if L >= H:
            return False
        eta = 2.0 * K[i, j] - K[i, i] - K[j, j]  # second derivative along the feasible line (≤ 0)
        if eta >= 0:
            return False
        a_j_old, a_i_old = a[j], a[i]
        a[j] = np.clip(a_j_old - y[j] * (E_i - E_j) / eta, L, H)  # Newton step on α_j, clipped to the box
        if abs(a[j] - a_j_old) < 1e-7:
            a[j] = a_j_old
            return False
        a[i] = a_i_old + y[i] * y[j] * (a_j_old - a[j])  # keep Σ α y = 0
        # Bias from the KKT condition f(x_i) = y_i for a free (0 < α < C) support vector.
        b1 = self.b - E_i - y[i] * (a[i] - a_i_old) * K[i, i] - y[j] * (a[j] - a_j_old) * K[i, j]
        b2 = self.b - E_j - y[i] * (a[i] - a_i_old) * K[i, j] - y[j] * (a[j] - a_j_old) * K[j, j]
        if 0 < a[i] < C:
            self.b = b1
        elif 0 < a[j] < C:
            self.b = b2
        else:
            self.b = 0.5 * (b1 + b2)
        return True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        N = X.shape[0]
        self.X, self.y = X, y.astype(float)
        self.alpha = np.zeros(N)  # (N,)
        self.b = 0.0
        K = self.kernel(X, X)  # (N, N)
        passes = 0
        while passes < self.max_passes:
            changed = 0
            for i in range(N):
                E_i = self._f(K, i) - self.y[i]
                violates = (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (
                    self.y[i] * E_i > self.tol and self.alpha[i] > 0
                )
                if not violates:
                    continue
                j = int(self.rng.integers(N - 1))
                j = j + 1 if j >= i else j  # any index ≠ i
                changed += int(self._update_pair(K, i, j))
            passes = passes + 1 if changed == 0 else 0
        return self

    @property
    def support_(self) -> np.ndarray:
        """Indices with ``α_i > 0``."""
        return np.flatnonzero(self.alpha > 1e-8)

    def decision_function(self, Q: np.ndarray) -> np.ndarray:
        sv = self.support_  # (S,)
        K = self.kernel(Q, self.X[sv])  # (M, S)
        return K @ (self.alpha[sv] * self.y[sv]) + self.b  # (M,)

    def predict(self, Q: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(Q) >= 0, 1, -1)  # (M,)


def dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """``Σ α_i - ½ Σ_ij α_i α_j y_i y_j K_ij``. ``alpha``, ``y``: (N,), ``K``: (N, N)."""
    ay = alpha * y  # (N,)
    return float(alpha.sum() - 0.5 * ay @ K @ ay)


def hinge_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, C: float) -> float:
    """Primal soft-margin objective ``½||w||² + C Σ_i max(0, 1 - y_i (w·x_i + b))``."""
    margins = y * (X @ w + b)  # (N,)
    return float(0.5 * w @ w + C * np.maximum(0.0, 1.0 - margins).sum())


def fit_linear_svm_primal(X: np.ndarray, y: np.ndarray, C: float = 1.0, lr: float = 1e-3, n_steps: int = 5000) -> tuple[np.ndarray, float]:
    """Subgradient descent on the primal hinge objective (the route that scales to huge N).

    Subgradient: ``w - C Σ_{i: margin_i < 1} y_i x_i``; ``-C Σ_{i: margin_i < 1} y_i`` for ``b``.
    ``X``: (N, d), ``y``: (N,) in {-1,+1} -> ``(w (d,), b)``.
    """
    w = np.zeros(X.shape[1])  # (d,)
    b = 0.0
    for _ in range(n_steps):
        margins = y * (X @ w + b)  # (N,)
        active = margins < 1.0  # (N,) bool: examples inside the margin or misclassified
        grad_w = w - C * (y[active, None] * X[active]).sum(axis=0)  # (d,)
        grad_b = -C * y[active].sum()
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, float(b)
