"""k-means: Lloyd's algorithm, k-means++ seeding, and mini-batch k-means.

Objective: ``J(C, μ) = Σ_i ||x_i - μ_{c_i}||²``. Lloyd alternates
(1) assign each point to its nearest centroid (minimises J over ``c`` with ``μ``
fixed) and (2) move each centroid to the mean of its points (minimises J over
``μ`` with ``c`` fixed). Both steps can only lower J, and there are finitely many
assignments, so the algorithm terminates at a local minimum.
"""

from __future__ import annotations

import numpy as np


def squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``||x_i - c_j||²`` for all pairs. ``X``: (N, d), ``C``: (k, d) -> (N, k)."""
    x2 = (X * X).sum(axis=1, keepdims=True)  # (N, 1)
    c2 = (C * C).sum(axis=1)[None, :]  # (1, k)
    return np.maximum(x2 + c2 - 2.0 * X @ C.T, 0.0)  # (N, k)


def kmeans_objective(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    """``Σ_i ||x_i - C[labels_i]||²``. ``X``: (N, d), ``C``: (k, d), ``labels``: (N,)."""
    diff = X - C[labels]  # (N, d)
    return float((diff * diff).sum())


def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii 2007).

    First centre uniform at random; each next centre is drawn with probability
    proportional to ``D(x)²``, its squared distance to the nearest centre chosen
    so far. The expected objective is within ``O(log k)`` of optimal.

    ``X``: (N, d) -> ``C``: (k, d).
    """
    N, d = X.shape
    C = np.empty((k, d))  # (k, d)
    C[0] = X[rng.integers(N)]
    d2 = squared_distances(X, C[:1]).min(axis=1)  # (N,) D(x)² to nearest chosen centre
    for j in range(1, k):
        probs = d2 / d2.sum()  # (N,)
        C[j] = X[rng.choice(N, p=probs)]
        d2 = np.minimum(d2, squared_distances(X, C[j : j + 1])[:, 0])  # (N,)
    return C


def kmeans(
    X: np.ndarray, k: int, n_iters: int = 100, init: str = "k-means++", seed: int = 0, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Lloyd's algorithm. ``X``: (N, d) -> ``(C (k, d), labels (N,), objective history)``.

    The history is non-increasing (each half-step is an exact coordinate minimisation).
    """
    rng = np.random.default_rng(seed)
    if init == "k-means++":
        C = kmeans_plusplus_init(X, k, rng)  # (k, d)
    else:
        C = X[rng.choice(X.shape[0], size=k, replace=False)]  # (k, d) random rows
    history: list[float] = []
    labels = np.zeros(X.shape[0], dtype=int)  # (N,)
    for _ in range(n_iters):
        labels = squared_distances(X, C).argmin(axis=1)  # (N,)   assignment step
        history.append(kmeans_objective(X, C, labels))
        new_C = C.copy()
        for j in range(k):
            members = X[labels == j]  # (N_j, d)
            if len(members) > 0:
                new_C[j] = members.mean(axis=0)  # (d,)  update step
        shift = float(((new_C - C) ** 2).sum())
        C = new_C
        if shift < tol:
            break
    labels = squared_distances(X, C).argmin(axis=1)  # (N,)
    history.append(kmeans_objective(X, C, labels))
    return C, labels, history


def minibatch_kmeans(
    X: np.ndarray, k: int, batch_size: int = 256, n_steps: int = 200, seed: int = 0
) -> np.ndarray:
    """Mini-batch k-means (Sculley 2010): per-centre learning rate ``1 / n_j``.

    Each step draws a batch, assigns it, and moves each centre toward its
    batch members with step size ``1 / (count so far)`` — a streaming mean.

    ``X``: (N, d) -> ``C``: (k, d).
    """
    rng = np.random.default_rng(seed)
    C = kmeans_plusplus_init(X, k, rng)  # (k, d)
    counts = np.zeros(k)  # (k,) points seen per centre
    for _ in range(n_steps):
        batch = X[rng.choice(X.shape[0], size=min(batch_size, X.shape[0]), replace=False)]  # (B, d)
        labels = squared_distances(batch, C).argmin(axis=1)  # (B,)
        for x, j in zip(batch, labels):
            counts[j] += 1
            eta = 1.0 / counts[j]
            C[j] = (1.0 - eta) * C[j] + eta * x  # (d,)
    return C
