# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/toy_embedder.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k toy_embedder -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/toy_embedder --force

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

    def __init__(self, dim: int=256, n: int=3) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _ngrams(self, text: str) -> list[str]:
        raise NotImplementedError('TODO: implement _ngrams (see the reference in src/mlbook)')

    def embed_one(self, text: str) -> np.ndarray:
        """One string -> (dim,) unit vector (zero vector if the text is empty)."""
        raise NotImplementedError('TODO: implement embed_one (see the reference in src/mlbook)')

    def embed(self, texts: list[str]) -> np.ndarray:
        """list of N strings -> (N, dim)."""
        raise NotImplementedError('TODO: implement embed (see the reference in src/mlbook)')
