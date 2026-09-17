# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/linear_regression.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k linear_regression -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/linear_regression --force

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
    raise NotImplementedError('TODO: implement add_bias (see the reference in src/mlbook)')

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return ``X @ w``. ``X``: (N, d), ``w``: (d,) -> (N,)."""
    raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

def mse_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Mean squared error ``(1/N) ||Xw - y||^2``. ``X``: (N, d), ``y``: (N,), ``w``: (d,)."""
    raise NotImplementedError('TODO: implement mse_loss (see the reference in src/mlbook)')

def fit_ols_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the normal equations ``(X^T X) w = X^T y`` with a linear solver.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    Requires ``X^T X`` to be invertible (``rank(X) = d``); otherwise use
    :func:`fit_ols_pinv`.
    """
    raise NotImplementedError('TODO: implement fit_ols_normal_equations (see the reference in src/mlbook)')

def fit_ols_pinv(X: np.ndarray, y: np.ndarray, rcond: float=1e-10) -> np.ndarray:
    """Minimum-norm least-squares solution ``w = X^+ y`` via the SVD.

    ``X = U S V^T`` (thin), ``X^+ = V S^+ U^T`` where ``S^+`` inverts the non-zero
    singular values and zeroes the rest. Works when ``X^T X`` is singular.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    raise NotImplementedError('TODO: implement fit_ols_pinv (see the reference in src/mlbook)')

def ols_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of ``L(w) = (1/N) ||Xw - y||^2``: ``(2/N) X^T (Xw - y)``.

    ``X``: (N, d), ``y``: (N,), ``w``: (d,) -> (d,).
    """
    raise NotImplementedError('TODO: implement ols_gradient (see the reference in src/mlbook)')

def fit_ols_gradient_descent(X: np.ndarray, y: np.ndarray, lr: float | None=None, n_steps: int=1000) -> np.ndarray:
    """Full-batch gradient descent on the MSE.

    If ``lr`` is ``None`` it is set to ``1 / L`` where ``L = 2 λ_max(X^T X) / N``
    is the Lipschitz constant of the gradient, which guarantees monotone descent.
    Convergence rate is governed by the condition number ``λ_max / λ_min`` of
    ``X^T X``: the error along the flattest direction contracts by
    ``(1 - λ_min/λ_max)`` per step.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    raise NotImplementedError('TODO: implement fit_ols_gradient_descent (see the reference in src/mlbook)')

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge: ``w = (X^T X + λ I)^{-1} X^T y``.

    ``X``: (N, d), ``y``: (N,), ``lam`` >= 0 -> ``w``: (d,).
    For ``lam > 0`` the system is always non-singular.
    """
    raise NotImplementedError('TODO: implement fit_ridge (see the reference in src/mlbook)')

def fit_ridge_svd(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge through the SVD, exposing the per-direction shrinkage.

    With ``X = U S V^T``: ``w = V diag(s_i / (s_i^2 + λ)) U^T y``. Direction ``i``
    is shrunk by the factor ``s_i^2 / (s_i^2 + λ)`` relative to OLS.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    raise NotImplementedError('TODO: implement fit_ridge_svd (see the reference in src/mlbook)')

def soft_threshold(z: np.ndarray, tau: float) -> np.ndarray:
    """Proximal operator of ``τ|·|``: ``sign(z) max(|z| - τ, 0)``. Elementwise."""
    raise NotImplementedError('TODO: implement soft_threshold (see the reference in src/mlbook)')

def fit_lasso_coordinate_descent(X: np.ndarray, y: np.ndarray, lam: float, n_sweeps: int=200, tol: float=1e-08) -> np.ndarray:
    """Lasso by cyclic coordinate descent.

    Objective: ``(1/(2N)) ||Xw - y||^2 + λ ||w||_1`` (glmnet convention).
    For coordinate ``j`` with all others fixed, the exact minimiser is

        w_j = S( (1/N) x_j^T r_{-j}, λ ) / ( (1/N) x_j^T x_j ),

    where ``r_{-j} = y - X w + x_j w_j`` is the partial residual excluding
    feature ``j`` and ``S`` is :func:`soft_threshold`.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    raise NotImplementedError('TODO: implement fit_lasso_coordinate_descent (see the reference in src/mlbook)')

def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """``(1/(2N)) ||Xw - y||^2 + λ ||w||_1``."""
    raise NotImplementedError('TODO: implement lasso_objective (see the reference in src/mlbook)')
