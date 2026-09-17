"""Softmax (multinomial logistic) regression from scratch.

Key results implemented here:

* ``softmax(z)_k = exp(z_k - m) / Σ_j exp(z_j - m)`` with ``m = max_j z_j``
  (log-sum-exp stabilisation);
* ``∂L/∂z = p - y`` for cross-entropy on top of softmax (the Jacobian of
  softmax is ``diag(p) - p p^T`` and it collapses when chained with ``-log p_y``);
* temperature ``softmax(z / T)`` and label smoothing ``y_ls = (1-ε) y + ε / K``.

Conventions: logits ``Z = X W`` with ``X`` (N, d), ``W`` (d, K), ``Z`` (N, K).
"""

from __future__ import annotations

import numpy as np


def log_softmax(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """``log softmax(Z / T)`` row-wise, computed as ``z - logsumexp(z)``.

    ``Z``: (N, K) -> (N, K).
    """
    Zt = Z / temperature  # (N, K)
    m = Zt.max(axis=1, keepdims=True)  # (N, 1)
    lse = m + np.log(np.exp(Zt - m).sum(axis=1, keepdims=True))  # (N, 1)
    return Zt - lse  # (N, K)


def softmax(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with max-subtraction. ``Z``: (N, K) -> (N, K), rows sum to 1."""
    return np.exp(log_softmax(Z, temperature))  # (N, K)


def softmax_jacobian(p: np.ndarray) -> np.ndarray:
    """Jacobian ``∂p_i/∂z_j = p_i (δ_ij - p_j)`` for one example.

    ``p``: (K,) -> (K, K), equals ``diag(p) - p p^T``.
    """
    return np.diag(p) - np.outer(p, p)  # (K, K)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """``y``: (N,) integer labels -> (N, K) one-hot."""
    Y = np.zeros((len(y), n_classes))  # (N, K)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def smooth_labels(Y: np.ndarray, eps: float) -> np.ndarray:
    """Label smoothing ``(1 - ε) Y + ε / K``. ``Y``: (N, K) -> (N, K)."""
    K = Y.shape[1]
    return (1.0 - eps) * Y + eps / K  # (N, K)


def cross_entropy(Z: np.ndarray, Y: np.ndarray, temperature: float = 1.0) -> float:
    """Mean cross-entropy ``-(1/N) Σ_i Σ_k Y_ik log p_ik``.

    ``Z``: (N, K) logits, ``Y``: (N, K) (one-hot or smoothed targets).
    """
    logp = log_softmax(Z, temperature)  # (N, K)
    return float(-(Y * logp).sum() / Z.shape[0])


def cross_entropy_grad_logits(Z: np.ndarray, Y: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """``∂L/∂Z = (1/N) (p - Y) / T``. ``Z``: (N, K), ``Y``: (N, K) -> (N, K).

    With smoothed labels the gradient is ``p - y_ls``: the model is pushed
    toward ``ε/K`` on the wrong classes instead of exactly zero, so logits stop
    growing without bound.
    """
    p = softmax(Z, temperature)  # (N, K)
    return (p - Y) / (temperature * Z.shape[0])  # (N, K)


def softmax_regression_loss(X: np.ndarray, Y: np.ndarray, W: np.ndarray, l2: float = 0.0) -> float:
    """Cross-entropy of ``softmax(XW)`` plus ``(l2/2) ||W||_F^2``."""
    Z = X @ W  # (N, K)
    return cross_entropy(Z, Y) + 0.5 * l2 * float((W * W).sum())


def softmax_regression_grad(X: np.ndarray, Y: np.ndarray, W: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """``∇_W L = (1/N) X^T (P - Y) + l2 W``. ``X``: (N, d), ``Y``: (N, K), ``W``: (d, K) -> (d, K)."""
    Z = X @ W  # (N, K)
    dZ = cross_entropy_grad_logits(Z, Y)  # (N, K)   (P - Y)/N
    return X.T @ dZ + l2 * W  # (d, K)


def fit_softmax_gd(
    X: np.ndarray, y: np.ndarray, n_classes: int, lr: float = 0.5, n_steps: int = 2000,
    l2: float = 0.0, label_smoothing: float = 0.0,
) -> np.ndarray:
    """Full-batch gradient descent for softmax regression.

    ``X``: (N, d), ``y``: (N,) integer labels -> ``W``: (d, K).
    """
    Y = one_hot(y, n_classes)  # (N, K)
    if label_smoothing > 0:
        Y = smooth_labels(Y, label_smoothing)  # (N, K)
    W = np.zeros((X.shape[1], n_classes))  # (d, K)
    for _ in range(n_steps):
        W = W - lr * softmax_regression_grad(X, Y, W, l2)  # (d, K)
    return W


def predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Argmax class. ``X``: (N, d), ``W``: (d, K) -> (N,)."""
    return (X @ W).argmax(axis=1)  # (N,)
