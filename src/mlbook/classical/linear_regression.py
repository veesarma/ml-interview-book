"""Linear regression: OLS (normal equations, pseudoinverse, gradient descent),
Ridge (closed form and SVD shrinkage view) and Lasso (coordinate descent with
soft-thresholding).

Conventions
-----------
* ``X`` is ``(N, d)``: one example per row. A bias column is *not* added
  automatically; call :func:`add_bias` when you want an intercept.
* ``y`` is ``(N,)``.
* Every estimator returns a weight vector ``w`` of shape ``(d,)`` such that the
  prediction is ``X @ w``.
"""

from __future__ import annotations

import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Append a column of ones so the last weight acts as the intercept.

    Input ``X``: (N, d). Output: (N, d + 1).
    """
    ones = np.ones((X.shape[0], 1))  # (N, 1)
    return np.concatenate([X, ones], axis=1)  # (N, d+1)


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return ``X @ w``. ``X``: (N, d), ``w``: (d,) -> (N,)."""
    return X @ w  # (N,)


def mse_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Mean squared error ``(1/N) ||Xw - y||^2``. ``X``: (N, d), ``y``: (N,), ``w``: (d,)."""
    r = X @ w - y  # (N,)
    return float(r @ r / X.shape[0])


# --------------------------------------------------------------------------- #
# Ordinary least squares
# --------------------------------------------------------------------------- #
def fit_ols_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the normal equations ``(X^T X) w = X^T y`` with a linear solver.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    Requires ``X^T X`` to be invertible (``rank(X) = d``); otherwise use
    :func:`fit_ols_pinv`.
    """
    gram = X.T @ X  # (d, d)
    rhs = X.T @ y  # (d,)
    return np.linalg.solve(gram, rhs)  # (d,)


def fit_ols_pinv(X: np.ndarray, y: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Minimum-norm least-squares solution ``w = X^+ y`` via the SVD.

    ``X = U S V^T`` (thin), ``X^+ = V S^+ U^T`` where ``S^+`` inverts the non-zero
    singular values and zeroes the rest. Works when ``X^T X`` is singular.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)  # U: (N, r), s: (r,), Vt: (r, d), r=min(N,d)
    s_inv = np.where(s > rcond * s.max(), 1.0 / np.maximum(s, 1e-300), 0.0)  # (r,)
    coeffs = s_inv * (U.T @ y)  # (r,)   S^+ U^T y
    return Vt.T @ coeffs  # (d,)   V S^+ U^T y


def ols_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of ``L(w) = (1/N) ||Xw - y||^2``: ``(2/N) X^T (Xw - y)``.

    ``X``: (N, d), ``y``: (N,), ``w``: (d,) -> (d,).
    """
    residual = X @ w - y  # (N,)
    return (2.0 / X.shape[0]) * (X.T @ residual)  # (d,)


def fit_ols_gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float | None = None, n_steps: int = 1000
) -> np.ndarray:
    """Full-batch gradient descent on the MSE.

    If ``lr`` is ``None`` it is set to ``1 / L`` where ``L = 2 λ_max(X^T X) / N``
    is the Lipschitz constant of the gradient, which guarantees monotone descent.
    Convergence rate is governed by the condition number ``λ_max / λ_min`` of
    ``X^T X``: the error along the flattest direction contracts by
    ``(1 - λ_min/λ_max)`` per step.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    N, d = X.shape
    if lr is None:
        lam_max = np.linalg.eigvalsh(X.T @ X)[-1]  # scalar, largest eigenvalue of (d, d)
        lr = N / (2.0 * lam_max)
    w = np.zeros(d)  # (d,)
    for _ in range(n_steps):
        w = w - lr * ols_gradient(X, y, w)  # (d,)
    return w


# --------------------------------------------------------------------------- #
# Ridge
# --------------------------------------------------------------------------- #
def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge: ``w = (X^T X + λ I)^{-1} X^T y``.

    ``X``: (N, d), ``y``: (N,), ``lam`` >= 0 -> ``w``: (d,).
    For ``lam > 0`` the system is always non-singular.
    """
    d = X.shape[1]
    gram = X.T @ X + lam * np.eye(d)  # (d, d)
    return np.linalg.solve(gram, X.T @ y)  # (d,)


def fit_ridge_svd(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge through the SVD, exposing the per-direction shrinkage.

    With ``X = U S V^T``: ``w = V diag(s_i / (s_i^2 + λ)) U^T y``. Direction ``i``
    is shrunk by the factor ``s_i^2 / (s_i^2 + λ)`` relative to OLS.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)  # U: (N, r), s: (r,), Vt: (r, d)
    shrink = s / (s**2 + lam)  # (r,)  s_i / (s_i^2 + λ)
    return Vt.T @ (shrink * (U.T @ y))  # (d,)


# --------------------------------------------------------------------------- #
# Lasso
# --------------------------------------------------------------------------- #
def soft_threshold(z: np.ndarray, tau: float) -> np.ndarray:
    """Proximal operator of ``τ|·|``: ``sign(z) max(|z| - τ, 0)``. Elementwise."""
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


def fit_lasso_coordinate_descent(
    X: np.ndarray, y: np.ndarray, lam: float, n_sweeps: int = 200, tol: float = 1e-8
) -> np.ndarray:
    """Lasso by cyclic coordinate descent.

    Objective: ``(1/(2N)) ||Xw - y||^2 + λ ||w||_1`` (glmnet convention).
    For coordinate ``j`` with all others fixed, the exact minimiser is

        w_j = S( (1/N) x_j^T r_{-j}, λ ) / ( (1/N) x_j^T x_j ),

    where ``r_{-j} = y - X w + x_j w_j`` is the partial residual excluding
    feature ``j`` and ``S`` is :func:`soft_threshold`.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    N, d = X.shape
    w = np.zeros(d)  # (d,)
    col_sq = (X * X).sum(axis=0) / N  # (d,)  (1/N) x_j^T x_j
    residual = y.copy()  # (N,)  y - X w, kept up to date incrementally
    for _ in range(n_sweeps):
        max_change = 0.0
        for j in range(d):
            if col_sq[j] == 0.0:
                continue
            x_j = X[:, j]  # (N,)
            partial = residual + x_j * w[j]  # (N,)  r_{-j}
            rho = x_j @ partial / N  # scalar  (1/N) x_j^T r_{-j}
            w_new = soft_threshold(rho, lam) / col_sq[j]  # scalar
            residual = residual - x_j * (w_new - w[j])  # (N,)
            max_change = max(max_change, abs(w_new - w[j]))
            w[j] = w_new
        if max_change < tol:
            break
    return w


def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """``(1/(2N)) ||Xw - y||^2 + λ ||w||_1``."""
    r = X @ w - y  # (N,)
    return float(r @ r / (2 * X.shape[0]) + lam * np.abs(w).sum())
