"""k-nearest neighbours: distance metrics, brute-force search and a kd-tree.

Brute force costs ``O(N d)`` per query (plus ``O(N)`` for a partial sort);
a kd-tree is ``O(log N)`` per query in low dimension but degrades toward
brute force once ``d`` exceeds roughly 10–20 (see the chapter). Beyond that,
use approximate methods (IVF/HNSW; Part XIII).
"""

from __future__ import annotations

import numpy as np


def pairwise_distances(Q: np.ndarray, X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distances between every query and every stored point.

    ``Q``: (M, d), ``X``: (N, d) -> (M, N).
    euclidean: ``||q - x||`` computed as ``sqrt(||q||² + ||x||² - 2 q·x)``;
    manhattan: ``Σ |q_j - x_j|``; cosine: ``1 - q·x / (||q|| ||x||)``.
    """
    if metric == "euclidean":
        q2 = (Q * Q).sum(axis=1, keepdims=True)  # (M, 1)
        x2 = (X * X).sum(axis=1)[None, :]  # (1, N)
        sq = q2 + x2 - 2.0 * Q @ X.T  # (M, N)
        return np.sqrt(np.maximum(sq, 0.0))  # (M, N)
    if metric == "manhattan":
        return np.abs(Q[:, None, :] - X[None, :, :]).sum(axis=2)  # (M, N) via (M, N, d)
    if metric == "cosine":
        Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)  # (M, d)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)  # (N, d)
        return 1.0 - Qn @ Xn.T  # (M, N)
    raise ValueError(f"unknown metric {metric!r}")


def knn_indices(Q: np.ndarray, X: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """Indices of the ``k`` nearest stored points for each query.

    ``Q``: (M, d), ``X``: (N, d) -> (M, k), sorted by increasing distance.
    ``argpartition`` finds the k smallest in ``O(N)`` and only those are sorted.
    """
    D = pairwise_distances(Q, X, metric)  # (M, N)
    part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]  # (M, k) unsorted k smallest
    row = np.arange(Q.shape[0])[:, None]  # (M, 1)
    order = np.argsort(D[row, part], axis=1)  # (M, k)
    return part[row, order]  # (M, k)


class KNNClassifier:
    """Majority vote among the ``k`` nearest training points (optionally distance-weighted)."""

    def __init__(self, k: int = 5, metric: str = "euclidean", weighted: bool = False) -> None:
        self.k, self.metric, self.weighted = k, metric, weighted
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store the data: training is ``O(1)``. ``X``: (N, d), ``y``: (N,) ints."""
        self.X, self.y = X, y.astype(int)
        self.n_classes = int(self.y.max()) + 1
        return self

    def predict_proba(self, Q: np.ndarray) -> np.ndarray:
        """``Q``: (M, d) -> (M, K) vote fractions."""
        idx = knn_indices(Q, self.X, self.k, self.metric)  # (M, k)
        labels = self.y[idx]  # (M, k)
        if self.weighted:
            D = pairwise_distances(Q, self.X, self.metric)  # (M, N)
            w = 1.0 / (np.take_along_axis(D, idx, axis=1) + 1e-8)  # (M, k)
        else:
            w = np.ones_like(labels, dtype=float)  # (M, k)
        votes = np.zeros((Q.shape[0], self.n_classes))  # (M, K)
        for j in range(self.k):
            votes[np.arange(Q.shape[0]), labels[:, j]] += w[:, j]
        return votes / votes.sum(axis=1, keepdims=True)  # (M, K)

    def predict(self, Q: np.ndarray) -> np.ndarray:
        return self.predict_proba(Q).argmax(axis=1)  # (M,)


class KNNRegressor:
    """Mean of the ``k`` nearest targets. ``fit(X (N, d), y (N,))``; ``predict(Q (M, d)) -> (M,)``."""

    def __init__(self, k: int = 5, metric: str = "euclidean") -> None:
        self.k, self.metric = k, metric
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.X, self.y = X, y.astype(float)
        return self

    def predict(self, Q: np.ndarray) -> np.ndarray:
        idx = knn_indices(Q, self.X, self.k, self.metric)  # (M, k)
        return self.y[idx].mean(axis=1)  # (M,)


# --------------------------------------------------------------------------- #
# kd-tree (exact, Euclidean)
# --------------------------------------------------------------------------- #
class KDTree:
    """A minimal kd-tree for exact Euclidean nearest-neighbour queries.

    Build: split on the axis with the largest spread at the median, recursively.
    Query: descend to the leaf, then backtrack, pruning any subtree whose
    splitting plane is farther than the current best distance.
    """

    def __init__(self, X: np.ndarray, leaf_size: int = 8) -> None:
        self.X = X  # (N, d)
        self.leaf_size = leaf_size
        self.root = self._build(np.arange(X.shape[0]))

    def _build(self, idx: np.ndarray) -> dict:
        if len(idx) <= self.leaf_size:
            return {"leaf": True, "idx": idx}
        pts = self.X[idx]  # (n, d)
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        order = np.argsort(pts[:, axis])  # (n,)
        mid = len(idx) // 2
        return {
            "leaf": False, "axis": axis, "split": float(pts[order[mid], axis]),
            "left": self._build(idx[order[:mid]]), "right": self._build(idx[order[mid:]]),
        }

    def query(self, q: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """``q``: (d,) -> (distances (k,), indices (k,)) sorted ascending."""
        best_d = np.full(k, np.inf)  # (k,)
        best_i = np.full(k, -1)  # (k,)

        def visit(node: dict) -> None:
            nonlocal best_d, best_i
            if node["leaf"]:
                d = np.linalg.norm(self.X[node["idx"]] - q, axis=1)  # (n_leaf,)
                cand_d = np.concatenate([best_d, d])  # (k + n_leaf,)
                cand_i = np.concatenate([best_i, node["idx"]])  # (k + n_leaf,)
                keep = np.argsort(cand_d)[:k]  # (k,)
                best_d, best_i = cand_d[keep], cand_i[keep]
                return
            diff = q[node["axis"]] - node["split"]
            near, far = (node["left"], node["right"]) if diff <= 0 else (node["right"], node["left"])
            visit(near)
            if abs(diff) < best_d[-1]:  # the far half-space may still hold a closer point
                visit(far)

        visit(self.root)
        return best_d, best_i
