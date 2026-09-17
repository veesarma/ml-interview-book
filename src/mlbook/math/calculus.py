"""Gradients, Jacobians, Hessians, and a finite-difference checker (pure NumPy).

Layout convention (stated once, used everywhere): **denominator layout** --
the gradient of a scalar with respect to a tensor has the *same shape as the
tensor*. That is what ``param.grad`` is in PyTorch and what an optimizer needs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of a scalar function.

    ``(df/dx)_i ~ (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)``, error O(eps^2).

    Args:
        f: maps an array of shape ``x.shape`` to a float.
        x: any shape; the gradient has the same shape (denominator layout).
    Returns:
        array with the same shape as ``x``.
    """
    x = x.astype(np.float64)
    grad = np.zeros_like(x)  # same shape as x
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """``||a - b|| / max(||a||, ||b||, tiny)`` -- the standard gradient-check metric."""
    num = np.linalg.norm(a - b)
    den = max(np.linalg.norm(a), np.linalg.norm(b), 1e-12)
    return float(num / den)


def numerical_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Jacobian ``J[i, j] = d f_i / d x_j`` of ``f: R^n -> R^m`` by central differences.

    Args:
        f: maps (n,) to (m,).
        x: (n,).
    Returns:
        (m, n).
    """
    x = x.astype(np.float64)
    n = x.shape[0]
    m = f(x).shape[0]
    J = np.zeros((m, n))  # (m, n)
    for j in range(n):
        e = np.zeros(n)  # (n,)
        e[j] = eps
        J[:, j] = (f(x + e) - f(x - e)) / (2.0 * eps)  # (m,)
    return J


def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Hessian ``H[i, j] = d^2 f / dx_i dx_j`` by differentiating the numerical gradient.

    Args:
        f: maps (n,) to float.
        x: (n,).
    Returns:
        (n, n), symmetrised.
    """
    n = x.shape[0]
    H = np.zeros((n, n))  # (n, n)
    for i in range(n):
        e = np.zeros(n)  # (n,)
        e[i] = eps
        g_plus = numerical_gradient(f, x + e)  # (n,)
        g_minus = numerical_gradient(f, x - e)  # (n,)
        H[i, :] = (g_plus - g_minus) / (2.0 * eps)
    return 0.5 * (H + H.T)  # (n, n)


# ---------------------------------------------------------------------------
# Closed-form gradients derived in the chapter
# ---------------------------------------------------------------------------


def grad_quadratic_form(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``d/dx (x^T A x) = (A + A^T) x`` (equals ``2 A x`` when ``A`` is symmetric).

    Args:
        A: (d, d). x: (d,).
    Returns:
        (d,).
    """
    return (A + A.T) @ x  # (d,)


def grad_linear_least_squares(X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``d/dW ||X W - Y||_F^2 = 2 X^T (X W - Y)``.

    Args:
        X: (N, d). W: (d, k). Y: (N, k).
    Returns:
        (d, k), same shape as W.
    """
    R = X @ W - Y  # (N, k) residual
    return 2.0 * X.T @ R  # (d, N) @ (N, k) -> (d, k)


def softmax_backward(dA: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Backprop through a row-wise softmax ``A = softmax(S)``.

    Row-wise Jacobian is ``diag(a) - a a^T``, so
    ``dS = A * (dA - sum_j(dA * A, axis=-1, keepdims=True))``.

    Args:
        dA: (..., n) upstream gradient. A: (..., n) softmax output.
    Returns:
        (..., n) gradient with respect to the logits.
    """
    inner = np.sum(dA * A, axis=-1, keepdims=True)  # (..., 1) = a^T dA per row
    return A * (dA - inner)  # (..., n)


def attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, dict]:
    """``Y = softmax(Q K^T / sqrt(d_k)) V`` with a cache for the backward pass.

    Args:
        Q: (T, d_k). K: (T, d_k). V: (T, d_v).
    Returns:
        Y: (T, d_v); cache with Q, K, V, A.
    """
    d_k = Q.shape[1]
    S = Q @ K.T / np.sqrt(d_k)  # (T, T)
    S = S - S.max(axis=-1, keepdims=True)  # (T, T) numerically stable
    A = np.exp(S)  # (T, T)
    A = A / A.sum(axis=-1, keepdims=True)  # (T, T) rows sum to 1
    Y = A @ V  # (T, d_v)
    return Y, {"Q": Q, "K": K, "V": V, "A": A}


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
    Q, K, V, A = cache["Q"], cache["K"], cache["V"], cache["A"]
    d_k = Q.shape[1]
    dV = A.T @ dY  # (T, T)^T @ (T, d_v) -> (T, d_v)
    dA = dY @ V.T  # (T, d_v) @ (d_v, T) -> (T, T)
    dS = softmax_backward(dA, A)  # (T, T)
    dQ = dS @ K / np.sqrt(d_k)  # (T, T) @ (T, d_k) -> (T, d_k)
    dK = dS.T @ Q / np.sqrt(d_k)  # (T, T) @ (T, d_k) -> (T, d_k)
    return dQ, dK, dV


def taylor_second_order(f: Callable[[np.ndarray], float], x0: np.ndarray, x: np.ndarray) -> float:
    """Second-order Taylor approximation of ``f`` around ``x0`` evaluated at ``x``.

    ``f(x) ~ f(x0) + g^T (x - x0) + 1/2 (x - x0)^T H (x - x0)`` with
    numerical ``g`` and ``H``. Used to explain Newton steps and curvature.

    Args:
        x0, x: (n,).
    """
    delta = x - x0  # (n,)
    g = numerical_gradient(f, x0)  # (n,)
    H = numerical_hessian(f, x0)  # (n, n)
    return float(f(x0) + g @ delta + 0.5 * delta @ H @ delta)
