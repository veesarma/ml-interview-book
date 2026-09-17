"""Binary logistic regression from scratch: sigmoid, Bernoulli negative
log-likelihood, gradient ``X^T (p - y)``, Hessian ``X^T diag(p(1-p)) X``,
gradient descent, Newton / IRLS, class weighting, and Platt scaling.

Conventions: ``X`` is ``(N, d)`` (add a bias column yourself), ``y`` is ``(N,)``
with entries in ``{0, 1}``, ``w`` is ``(d,)``.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``σ(z) = 1 / (1 + e^{-z})``, elementwise.

    For ``z >= 0`` use ``1 / (1 + e^{-z})``; for ``z < 0`` use ``e^{z} / (1 + e^{z})``
    so the exponent is never large and positive.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    """``log σ(z) = -softplus(-z) = -log(1 + e^{-z})`` computed stably."""
    return -np.logaddexp(0.0, -z)


def bce_loss(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, sample_weight: np.ndarray | None = None, l2: float = 0.0
) -> float:
    """Mean binary cross-entropy (negative Bernoulli log-likelihood) plus ``(l2/2)||w||^2``.

    ``L = -(1/N) Σ_i s_i [ y_i log σ(z_i) + (1 - y_i) log(1 - σ(z_i)) ]`` with ``z = Xw``.
    ``X``: (N, d), ``y``: (N,), ``w``: (d,), ``sample_weight``: (N,) or None.
    """
    z = X @ w  # (N,)
    s = np.ones_like(y, dtype=float) if sample_weight is None else sample_weight  # (N,)
    per_example = -(y * log_sigmoid(z) + (1.0 - y) * log_sigmoid(-z))  # (N,)
    return float((s * per_example).sum() / X.shape[0] + 0.5 * l2 * (w @ w))


def bce_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, sample_weight: np.ndarray | None = None, l2: float = 0.0
) -> np.ndarray:
    """``∇_w L = (1/N) X^T (s ⊙ (p - y)) + l2 w`` with ``p = σ(Xw)``.

    ``X``: (N, d), ``y``: (N,), ``w``: (d,) -> (d,).
    """
    p = sigmoid(X @ w)  # (N,)
    s = np.ones_like(y, dtype=float) if sample_weight is None else sample_weight  # (N,)
    return X.T @ (s * (p - y)) / X.shape[0] + l2 * w  # (d,)


def bce_hessian(
    X: np.ndarray, w: np.ndarray, sample_weight: np.ndarray | None = None, l2: float = 0.0
) -> np.ndarray:
    """``H = (1/N) X^T diag(s ⊙ p ⊙ (1 - p)) X + l2 I``. PSD, so the loss is convex.

    ``X``: (N, d), ``w``: (d,) -> (d, d).
    """
    p = sigmoid(X @ w)  # (N,)
    s = np.ones(X.shape[0]) if sample_weight is None else sample_weight  # (N,)
    r = s * p * (1.0 - p)  # (N,)  per-example curvature
    return (X * r[:, None]).T @ X / X.shape[0] + l2 * np.eye(X.shape[1])  # (d, d)


def fit_logistic_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_steps: int = 2000,
    l2: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Full-batch gradient descent on the (weighted, L2-regularised) BCE.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    w = np.zeros(X.shape[1])  # (d,)
    for _ in range(n_steps):
        w = w - lr * bce_gradient(X, y, w, sample_weight, l2)  # (d,)
    return w


def fit_logistic_newton(
    X: np.ndarray,
    y: np.ndarray,
    n_steps: int = 20,
    l2: float = 1e-6,
    sample_weight: np.ndarray | None = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Newton's method (equivalently IRLS): ``w ← w - H^{-1} ∇L``.

    Each step solves a weighted least-squares problem with weights
    ``p(1-p)`` and working response ``z + (y - p) / (p(1-p))``. Quadratic
    convergence near the optimum; ``O(N d^2 + d^3)`` per step.
    A small ``l2`` keeps ``H`` invertible when the data are separable.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    w = np.zeros(X.shape[1])  # (d,)
    for _ in range(n_steps):
        g = bce_gradient(X, y, w, sample_weight, l2)  # (d,)
        H = bce_hessian(X, w, sample_weight, l2)  # (d, d)
        step = np.linalg.solve(H, g)  # (d,)
        w = w - step  # (d,)
        if np.abs(step).max() < tol:
            break
    return w


def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    """Per-example weights ``N / (2 N_c)`` so each class contributes equally.

    ``y``: (N,) in {0,1} -> (N,).
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    w_pos = len(y) / (2.0 * max(n_pos, 1))
    w_neg = len(y) / (2.0 * max(n_neg, 1))
    return np.where(y == 1, w_pos, w_neg)  # (N,)


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``σ(Xw)``. ``X``: (N, d), ``w``: (d,) -> (N,)."""
    return sigmoid(X @ w)


def predict_label(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold the probability. Choose ``threshold`` from the cost matrix, not 0.5 by reflex."""
    return (predict_proba(X, w) >= threshold).astype(int)  # (N,)


def platt_scaling(scores: np.ndarray, y: np.ndarray, n_steps: int = 50) -> tuple[float, float]:
    """Fit ``P(y=1 | s) = σ(a s + b)`` by Newton's method. Platt scaling *is*
    1-D logistic regression on the model's raw score.

    ``scores``: (N,), ``y``: (N,) in {0,1} -> ``(a, b)``.
    """
    Xs = np.stack([scores, np.ones_like(scores)], axis=1)  # (N, 2)
    w = fit_logistic_newton(Xs, y, n_steps=n_steps, l2=1e-8)  # (2,)
    return float(w[0]), float(w[1])


def one_vs_rest_fit(X: np.ndarray, y: np.ndarray, n_classes: int, **kwargs) -> np.ndarray:
    """Train ``K`` independent binary classifiers (class k vs the rest).

    ``X``: (N, d), ``y``: (N,) in {0..K-1} -> ``W``: (d, K).
    """
    W = np.zeros((X.shape[1], n_classes))  # (d, K)
    for k in range(n_classes):
        W[:, k] = fit_logistic_newton(X, (y == k).astype(float), **kwargs)  # (d,)
    return W


def one_vs_rest_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Argmax over the ``K`` sigmoid scores (they need not sum to 1).

    ``X``: (N, d), ``W``: (d, K) -> (N,).
    """
    scores = sigmoid(X @ W)  # (N, K)
    return scores.argmax(axis=1)  # (N,)
