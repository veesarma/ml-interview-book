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

    def __init__(self, n_subvectors: int = 8, n_codes: int = 256, n_iters: int = 15, seed: int = 0) -> None:
        self.M = n_subvectors
        self.K = n_codes
        self.n_iters = n_iters
        self.seed = seed
        self.codebooks: np.ndarray | None = None  # (M, K, d_sub)
        self.d_sub: int = 0

    def train(self, X: np.ndarray) -> "ProductQuantizer":
        """Learn one k-means codebook per sub-space from X (N, d)."""
        N, d = X.shape
        assert d % self.M == 0, "d must be divisible by n_subvectors"
        self.d_sub = d // self.M
        books = []
        for m in range(self.M):
            sub = X[:, m * self.d_sub : (m + 1) * self.d_sub]  # (N, d_sub)
            cents, _ = kmeans(sub, self.K, self.n_iters, self.seed + m)  # (K, d_sub)
            books.append(cents)
        self.codebooks = np.stack(books, axis=0)  # (M, K, d_sub)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X (N, d) -> codes (N, M) of ints in [0, K)."""
        assert self.codebooks is not None
        N = X.shape[0]
        codes = np.zeros((N, self.M), dtype=np.int64)  # (N, M)
        for m in range(self.M):
            sub = X[:, m * self.d_sub : (m + 1) * self.d_sub]  # (N, d_sub)
            d2 = squared_euclidean(sub, self.codebooks[m])  # (N, K)
            codes[:, m] = np.argmin(d2, axis=1)  # (N,)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """codes (N, M) -> reconstructed vectors (N, d)."""
        assert self.codebooks is not None
        parts = [self.codebooks[m][codes[:, m]] for m in range(self.M)]  # M x (N, d_sub)
        return np.concatenate(parts, axis=1)  # (N, d)

    def distance_table(self, q: np.ndarray) -> np.ndarray:
        """ADC look-up table for one query q (d,) -> (M, K)."""
        assert self.codebooks is not None
        table = np.zeros((self.M, self.K))  # (M, K)
        for m in range(self.M):
            q_sub = q[m * self.d_sub : (m + 1) * self.d_sub][None, :]  # (1, d_sub)
            table[m] = squared_euclidean(q_sub, self.codebooks[m])[0]  # (K,)
        return table

    def asymmetric_distances(self, q: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Approximate ||q - x_i||^2 for all encoded x_i. q (d,), codes (N, M) -> (N,)."""
        table = self.distance_table(q)  # (M, K)
        rows = np.arange(self.M)[None, :]  # (1, M)
        return table[rows, codes].sum(axis=1)  # (N,)  gather then sum over sub-spaces

    def search(self, q: np.ndarray, codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k by ADC. -> (ids (k,), approx d2 (k,))."""
        d2 = self.asymmetric_distances(q, codes)  # (N,)
        k = min(k, codes.shape[0])
        ids = np.argsort(d2)[:k]  # (k,)
        return ids, d2[ids]

    def bytes_per_vector(self) -> float:
        """Storage of one code: M * log2(K) / 8 bytes."""
        return self.M * np.log2(self.K) / 8.0
