# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/hnsw.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k hnsw -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/hnsw --force

"""A small Hierarchical Navigable Small World (HNSW) graph from scratch.

Malkov & Yashunin (2016). Each element gets a random top level
    l = floor(-ln(U) * mL),   U ~ Uniform(0, 1),   mL = 1 / ln(M)
so layer sizes decay geometrically. Insertion: greedy descent from the entry
point through layers above l (ef = 1), then at each layer <= l a beam search
with ef_construction candidates, connect to the M closest, prune neighbours to
M_max. Query: greedy descent to layer 0, beam search with ef >= k, return top-k.
Distances are squared L2.
"""
from __future__ import annotations
import heapq
import math
import numpy as np

class HNSW:
    """Minimal multi-layer HNSW index over vectors (d,)."""

    def __init__(self, M: int=8, ef_construction: int=64, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _dist(self, a: np.ndarray, i: int) -> float:
        raise NotImplementedError('TODO: implement _dist (see the reference in src/mlbook)')

    def _random_level(self) -> int:
        raise NotImplementedError('TODO: implement _random_level (see the reference in src/mlbook)')

    def _search_layer(self, q: np.ndarray, entry: int, ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search at one layer. Returns up to ef (dist, id) pairs, closest first."""
        raise NotImplementedError('TODO: implement _search_layer (see the reference in src/mlbook)')

    def _connect(self, new: int, neighbours: list[int], layer: int) -> None:
        """Add bidirectional links and prune any node exceeding its degree cap."""
        raise NotImplementedError('TODO: implement _connect (see the reference in src/mlbook)')

    def add(self, x: np.ndarray) -> int:
        """Insert one vector x (d,). Returns its id."""
        raise NotImplementedError('TODO: implement add (see the reference in src/mlbook)')

    def search(self, q: np.ndarray, k: int, ef: int=32) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN of q (d,). -> (ids (k,), squared distances (k,))."""
        raise NotImplementedError('TODO: implement search (see the reference in src/mlbook)')
