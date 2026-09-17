# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/ivf.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ivf -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/ivf --force

"""IVF (inverted file) approximate nearest-neighbour index from scratch (NumPy).

Build: run k-means on the corpus to get n_list centroids (the coarse quantiser),
assign each vector to its nearest centroid, store an inverted list per centroid.
Search: find the nprobe centroids closest to the query, scan only their lists
exactly. Cost per query ~ n_list * d + nprobe * (N / n_list) * d instead of N * d.
"""
from __future__ import annotations
import numpy as np
from mlbook.retrieval.similarity import squared_euclidean

def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: each new centre is drawn with probability proportional
    to its squared distance from the nearest centre chosen so far. X (N, d) -> (k, d)."""
    raise NotImplementedError('TODO: implement kmeans_pp_init (see the reference in src/mlbook)')

def kmeans(X: np.ndarray, k: int, n_iters: int=20, seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ initialisation.

    X (N, d) -> centroids (k, d), assignments (N,). Empty clusters are re-seeded
    with a random row so every centroid stays live.
    """
    raise NotImplementedError('TODO: implement kmeans (see the reference in src/mlbook)')

class IVFIndex:
    """IVF-Flat: coarse k-means quantiser + exact scan inside probed lists."""

    def __init__(self, n_list: int, n_iters: int=20, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def train(self, X: np.ndarray) -> 'IVFIndex':
        """Fit the coarse quantiser on X (N, d)."""
        raise NotImplementedError('TODO: implement train (see the reference in src/mlbook)')

    def add(self, X: np.ndarray) -> 'IVFIndex':
        """Assign every row of X (N, d) to its nearest centroid's inverted list."""
        raise NotImplementedError('TODO: implement add (see the reference in src/mlbook)')

    def search(self, q: np.ndarray, k: int, nprobe: int=1) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN of one query q (d,). Returns (ids (k',), d2 (k',)), k' <= k.

        Steps: rank centroids by distance, take the closest nprobe, concatenate their
        lists, scan those candidates exactly, return the k best.
        """
        raise NotImplementedError('TODO: implement search (see the reference in src/mlbook)')
