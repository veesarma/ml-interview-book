"""IVF (inverted file) approximate nearest-neighbour index from scratch (NumPy).

Build: run k-means on the corpus to get n_list centroids (the coarse quantiser),
assign each vector to its nearest centroid, store an inverted list per centroid.
Search: find the nprobe centroids closest to the query, scan only their lists
exactly. Cost per query ~ n_list * d + nprobe * (N / n_list) * d instead of N * d.
"""
from __future__ import annotations

import numpy as np

from mlbook.retrieval.similarity import squared_euclidean


def kmeans(
    X: np.ndarray, k: int, n_iters: int = 20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with random-row initialisation.

    X (N, d) -> centroids (k, d), assignments (N,). Empty clusters are re-seeded
    with a random row so every centroid stays live.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    centroids = X[rng.choice(N, size=k, replace=False)].copy()  # (k, d)
    assign = np.zeros(N, dtype=int)  # (N,)
    for _ in range(n_iters):
        d2 = squared_euclidean(X, centroids)  # (N, k)
        new_assign = np.argmin(d2, axis=1)  # (N,)
        if np.array_equal(new_assign, assign) and _ > 0:
            break
        assign = new_assign
        for j in range(k):
            members = X[assign == j]  # (n_j, d)
            if members.shape[0] == 0:
                centroids[j] = X[rng.integers(N)]  # (d,) re-seed empty cluster
            else:
                centroids[j] = members.mean(axis=0)  # (d,)
    return centroids, assign


class IVFIndex:
    """IVF-Flat: coarse k-means quantiser + exact scan inside probed lists."""

    def __init__(self, n_list: int, n_iters: int = 20, seed: int = 0) -> None:
        self.n_list = n_list
        self.n_iters = n_iters
        self.seed = seed
        self.centroids: np.ndarray | None = None  # (n_list, d)
        self.lists: list[np.ndarray] = []  # per centroid: ids (n_j,)
        self.X: np.ndarray | None = None  # (N, d)

    def train(self, X: np.ndarray) -> "IVFIndex":
        """Fit the coarse quantiser on X (N, d)."""
        self.centroids, _ = kmeans(X, self.n_list, self.n_iters, self.seed)  # (n_list, d)
        return self

    def add(self, X: np.ndarray) -> "IVFIndex":
        """Assign every row of X (N, d) to its nearest centroid's inverted list."""
        assert self.centroids is not None, "call train() first"
        self.X = X  # (N, d)
        d2 = squared_euclidean(X, self.centroids)  # (N, n_list)
        assign = np.argmin(d2, axis=1)  # (N,)
        self.lists = [np.flatnonzero(assign == j) for j in range(self.n_list)]
        return self

    def search(self, q: np.ndarray, k: int, nprobe: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN of one query q (d,). Returns (ids (k',), d2 (k',)), k' <= k.

        Steps: rank centroids by distance, take the closest nprobe, concatenate their
        lists, scan those candidates exactly, return the k best.
        """
        assert self.X is not None and self.centroids is not None
        q2 = q[None, :]  # (1, d)
        cd2 = squared_euclidean(q2, self.centroids)[0]  # (n_list,)
        probe = np.argsort(cd2)[:nprobe]  # (nprobe,)
        cand = np.concatenate([self.lists[j] for j in probe])  # (n_cand,)
        if cand.size == 0:
            return np.zeros(0, dtype=int), np.zeros(0)
        d2 = squared_euclidean(q2, self.X[cand])[0]  # (n_cand,)
        k = min(k, cand.size)
        order = np.argsort(d2)[:k]  # (k,)
        return cand[order], d2[order]
