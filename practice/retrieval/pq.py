# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/pq.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k pq -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/pq --force

"""Product quantisation (PQ) with asymmetric distance computation (NumPy).

Split each d-dim vector into M sub-vectors of d/M dims. Learn a codebook of K
centroids per sub-space with k-means. A vector is stored as M codes of log2(K)
bits: with K = 256 and M = 8 a 128-float vector (512 bytes) becomes 8 bytes.

Asymmetric distance (ADC): keep the query in full precision, precompute a table
    T[m, j] = || q_m - C_m[j] ||^2                        (M, K)
and approximate ||q - x||^2 ~= sum_m T[m, code_m(x)] with M table look-ups.
"""
from __future__ import annotations
import numpy as np
from mlbook.retrieval.ivf import kmeans
from mlbook.retrieval.similarity import squared_euclidean

class ProductQuantizer:
    """PQ encoder / decoder / ADC search. d must be divisible by n_subvectors."""

    def __init__(self, n_subvectors: int=8, n_codes: int=256, n_iters: int=15, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def train(self, X: np.ndarray) -> 'ProductQuantizer':
        """Learn one k-means codebook per sub-space from X (N, d)."""
        raise NotImplementedError('TODO: implement train (see the reference in src/mlbook)')

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X (N, d) -> codes (N, M) of ints in [0, K)."""
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """codes (N, M) -> reconstructed vectors (N, d)."""
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def distance_table(self, q: np.ndarray) -> np.ndarray:
        """ADC look-up table for one query q (d,) -> (M, K)."""
        raise NotImplementedError('TODO: implement distance_table (see the reference in src/mlbook)')

    def asymmetric_distances(self, q: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Approximate ||q - x_i||^2 for all encoded x_i. q (d,), codes (N, M) -> (N,)."""
        raise NotImplementedError('TODO: implement asymmetric_distances (see the reference in src/mlbook)')

    def search(self, q: np.ndarray, codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k by ADC. -> (ids (k,), approx d2 (k,))."""
        raise NotImplementedError('TODO: implement search (see the reference in src/mlbook)')

    def bytes_per_vector(self) -> float:
        """Storage of one code: M * log2(K) / 8 bytes."""
        raise NotImplementedError('TODO: implement bytes_per_vector (see the reference in src/mlbook)')
