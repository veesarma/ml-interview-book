"""A deterministic, dependency-free text embedder for offline tests.

Hashes character n-grams into a fixed-size vector (the 'hashing trick'), then
L2-normalises. Two strings sharing many character n-grams get a high cosine, so
the embedder behaves like a crude lexical-semantic model: enough to exercise a
retrieval pipeline end to end without a neural network.
"""
from __future__ import annotations

import zlib

import numpy as np


class HashNGramEmbedder:
    """Character n-gram hashing embedder. embed(list[str]) -> (N, dim), rows unit-norm."""

    def __init__(self, dim: int = 256, n: int = 3) -> None:
        self.dim = dim
        self.n = n

    def _ngrams(self, text: str) -> list[str]:
        t = f" {text.lower().strip()} "
        if len(t) < self.n:
            return [t]
        return [t[i : i + self.n] for i in range(len(t) - self.n + 1)]

    def embed_one(self, text: str) -> np.ndarray:
        """One string -> (dim,) unit vector (zero vector if the text is empty)."""
        v = np.zeros(self.dim)  # (dim,)
        for g in self._ngrams(text):
            h = zlib.crc32(g.encode("utf-8"))  # deterministic across processes
            sign = 1.0 if (h >> 31) & 1 else -1.0  # top bit chooses the sign
            v[h % self.dim] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        """list of N strings -> (N, dim)."""
        return np.stack([self.embed_one(t) for t in texts], axis=0)  # (N, dim)
