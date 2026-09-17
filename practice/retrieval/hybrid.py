# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/hybrid.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k hybrid -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/hybrid --force

"""Score fusion for hybrid (sparse + dense) retrieval.

Reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009):
    RRF(d) = sum_{r in rankings} 1 / (k + rank_r(d)),   rank starts at 1, k ~ 60.
Rank-based fusion needs no score normalisation, which is why it is the default
when BM25 scores (unbounded) meet cosine scores (in [-1, 1]).
"""
from __future__ import annotations
import numpy as np

def reciprocal_rank_fusion(rankings: list[list[int]], k: int=60) -> list[tuple[int, float]]:
    """Fuse ranked id lists. Returns [(id, fused_score), ...] sorted best first."""
    raise NotImplementedError('TODO: implement reciprocal_rank_fusion (see the reference in src/mlbook)')

def min_max_normalize(scores: np.ndarray, eps: float=1e-12) -> np.ndarray:
    """Map scores (N,) to [0, 1] so heterogeneous scores can be added."""
    raise NotImplementedError('TODO: implement min_max_normalize (see the reference in src/mlbook)')

def convex_fusion(dense: np.ndarray, sparse: np.ndarray, alpha: float=0.5) -> np.ndarray:
    """alpha * norm(dense) + (1 - alpha) * norm(sparse). Both (N,) -> (N,)."""
    raise NotImplementedError('TODO: implement convex_fusion (see the reference in src/mlbook)')
