# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/math/calculus.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k calculus -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py math/calculus --force

"""Gradients, Jacobians, Hessians, and a finite-difference checker (pure NumPy).

Layout convention (stated once, used everywhere): **denominator layout** --
the gradient of a scalar with respect to a tensor has the *same shape as the
tensor*. That is what ``param.grad`` is in PyTorch and what an optimizer needs.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float=1e-06) -> np.ndarray:
    """Central-difference gradient of a scalar function.

    ``(df/dx)_i ~ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)``, error O(eps^2).

    Args:
        f: maps an array of shape ``x.shape`` to a float.
        x: any shape; the gradient has the same shape (denominator layout).
    Returns:
        array with the same shape as ``x``.
    """
    raise NotImplementedError('TODO: implement numerical_gradient (see the reference in src/mlbook)')

def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """``||a - b|| / max(||a||, ||b||, tiny)`` -- the standard gradient-check metric."""
    raise NotImplementedError('TODO: implement relative_error (see the reference in src/mlbook)')

def numerical_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float=1e-06) -> np.ndarray:
    """Jacobian ``J[i, j] = d f_i / d x_j`` of ``f: R^n -> R^m`` by central differences.

    Args:
        f: maps (n,) to (m,).
        x: (n,).
    Returns:
        (m, n).
    """
    raise NotImplementedError('TODO: implement numerical_jacobian (see the reference in src/mlbook)')

def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float=0.0001) -> np.ndarray:
    """Hessian ``H[i, j] = d^2 f / dx_i dx_j`` by differentiating the numerical gradient.

    Args:
        f: maps (n,) to float.
        x: (n,).
    Returns:
        (n, n), symmetrised.
    """
    raise NotImplementedError('TODO: implement numerical_hessian (see the reference in src/mlbook)')

def grad_quadratic_form(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``d/dx (x^T A x) = (A + A^T) x`` (equals ``2 A x`` when ``A`` is symmetric).

    Args:
        A: (d, d). x: (d,).
    Returns:
        (d,).
    """
    raise NotImplementedError('TODO: implement grad_quadratic_form (see the reference in src/mlbook)')

def grad_linear_least_squares(X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``d/dW ||X W - Y||_F^2 = 2 X^T (X W - Y)``.

    Args:
        X: (N, d). W: (d, k). Y: (N, k).
    Returns:
        (d, k), same shape as W.
    """
    raise NotImplementedError('TODO: implement grad_linear_least_squares (see the reference in src/mlbook)')

def softmax_backward(dA: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Backprop through a row-wise softmax ``A = softmax(S)``.

    Row-wise Jacobian is ``diag(a) - a a^T``, so
    ``dS = A * (dA - sum_j(dA * A, axis=-1, keepdims=True))``.

    Args:
        dA: (..., n) upstream gradient. A: (..., n) softmax output.
    Returns:
        (..., n) gradient with respect to the logits.
    """
    raise NotImplementedError('TODO: implement softmax_backward (see the reference in src/mlbook)')

def attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, dict]:
    """``Y = softmax(Q K^T / sqrt(d_k)) V`` with a cache for the backward pass.

    Args:
        Q: (T, d_k). K: (T, d_k). V: (T, d_v).
    Returns:
        Y: (T, d_v); cache with Q, K, V, A.
    """
    raise NotImplementedError('TODO: implement attention_forward (see the reference in src/mlbook)')

def attention_backward(dY: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients ``dQ, dK, dV`` of attention given ``dY = dL/dY``.

    Steps (each line is a matrix-calculus identity from the chapter):
      ``dV = A^T dY``            (from Y = A V)
      ``dA = dY V^T``            (from Y = A V)
      ``dS = A * (dA - rowsum(dA * A))``  (softmax Jacobian)
      ``dQ = dS K / sqrt(d_k)``  (from S = Q K^T / sqrt(d_k))
      ``dK = dS^T Q / sqrt(d_k)``

    Args:
        dY: (T, d_v).
    Returns:
        dQ: (T, d_k), dK: (T, d_k), dV: (T, d_v).
    """
    raise NotImplementedError('TODO: implement attention_backward (see the reference in src/mlbook)')

def taylor_second_order(f: Callable[[np.ndarray], float], x0: np.ndarray, x: np.ndarray) -> float:
    """Second-order Taylor approximation of ``f`` around ``x0`` evaluated at ``x``.

    ``f(x) ~ f(x0) + g^T (x - x0) + 1/2 (x - x0)^T H (x - x0)`` with
    numerical ``g`` and ``H``. Used to explain Newton steps and curvature.

    Args:
        x0, x: (n,).
    """
    raise NotImplementedError('TODO: implement taylor_second_order (see the reference in src/mlbook)')
