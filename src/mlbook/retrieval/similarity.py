"""Similarity functions and the MIPS -> nearest-neighbour reduction (NumPy).

Conventions: row-major data, one vector per row.
    Q: (Nq, d) queries, X: (N, d) corpus.

Equations implemented
    dot(q, x)    = q . x
    cos(q, x)    = q . x / (||q|| ||x||)
    ||q - x||^2  = ||q||^2 + ||x||^2 - 2 q . x
When rows are L2-normalised all three orderings coincide:
    ||q - x||^2 = 2 - 2 cos(q, x).
"""
from __future__ import annotations

import numpy as np


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise each row. (N, d) -> (N, d)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # (N, 1)
    return X / np.maximum(norms, eps)  # (N, d)


def dot_product(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Inner products. Q (Nq, d), X (N, d) -> (Nq, N)."""
    return Q @ X.T  # (Nq, N)


def cosine_similarity(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosine similarity. Q (Nq, d), X (N, d) -> (Nq, N)."""
    Qn = normalize_rows(Q)  # (Nq, d)
    Xn = normalize_rows(X)  # (N, d)
    return Qn @ Xn.T  # (Nq, N)


def squared_euclidean(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Squared L2 distance via the expansion ||q||^2 + ||x||^2 - 2 q.x.

    Q (Nq, d), X (N, d) -> (Nq, N). Clipped at 0 to absorb rounding error.
    """
    q2 = np.sum(Q * Q, axis=1, keepdims=True)  # (Nq, 1)
    x2 = np.sum(X * X, axis=1)[None, :]  # (1, N)
    d2 = q2 + x2 - 2.0 * (Q @ X.T)  # (Nq, N)
    return np.maximum(d2, 0.0)  # (Nq, N)


def mips_to_nn_corpus(X: np.ndarray) -> np.ndarray:
    """Augment the corpus so that inner-product search becomes L2 search.

    Shrivastava & Li (2014) / Bachrach et al. (2014): with M = max_i ||x_i||,
        x' = [x, sqrt(M^2 - ||x||^2)]      (all x' have norm M)
        q' = [q, 0]
        ||q' - x'||^2 = ||q||^2 + M^2 - 2 q.x
    so argmin_x ||q' - x'|| = argmax_x q.x.   X (N, d) -> (N, d + 1).
    """
    norms2 = np.sum(X * X, axis=1)  # (N,)
    M2 = float(np.max(norms2))
    extra = np.sqrt(np.maximum(M2 - norms2, 0.0))[:, None]  # (N, 1)
    return np.concatenate([X, extra], axis=1)  # (N, d + 1)


def mips_to_nn_query(Q: np.ndarray) -> np.ndarray:
    """Append a zero coordinate to each query. Q (Nq, d) -> (Nq, d + 1)."""
    zeros = np.zeros((Q.shape[0], 1), dtype=Q.dtype)  # (Nq, 1)
    return np.concatenate([Q, zeros], axis=1)  # (Nq, d + 1)


def brute_force_topk(
    Q: np.ndarray, X: np.ndarray, k: int, metric: str = "dot"
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by scanning the whole corpus.

    metric in {"dot", "cosine", "l2"}. Returns (ids (Nq, k), scores (Nq, k)) ordered
    best-first; for "l2" the score is the squared distance (smaller is better).
    """
    if metric == "dot":
        S = dot_product(Q, X)  # (Nq, N)
    elif metric == "cosine":
        S = cosine_similarity(Q, X)  # (Nq, N)
    elif metric == "l2":
        S = -squared_euclidean(Q, X)  # (Nq, N) negate so larger is better
    else:
        raise ValueError(f"unknown metric {metric}")
    k = min(k, X.shape[0])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]  # (Nq, k) unordered top-k
    part_scores = np.take_along_axis(S, part, axis=1)  # (Nq, k)
    order = np.argsort(-part_scores, axis=1)  # (Nq, k)
    ids = np.take_along_axis(part, order, axis=1)  # (Nq, k)
    scores = np.take_along_axis(part_scores, order, axis=1)  # (Nq, k)
    if metric == "l2":
        scores = -scores  # (Nq, k) back to distances
    return ids, scores
