"""Score fusion for hybrid (sparse + dense) retrieval.

Reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009):
    RRF(d) = sum_{r in rankings} 1 / (k + rank_r(d)),   rank starts at 1, k ~ 60.
Rank-based fusion needs no score normalisation, which is why it is the default
when BM25 scores (unbounded) meet cosine scores (in [-1, 1]).
"""
from __future__ import annotations

import numpy as np


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked id lists. Returns [(id, fused_score), ...] sorted best first."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def min_max_normalize(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map scores (N,) to [0, 1] so heterogeneous scores can be added."""
    lo, hi = float(scores.min()), float(scores.max())
    return (scores - lo) / max(hi - lo, eps)  # (N,)


def convex_fusion(dense: np.ndarray, sparse: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """alpha * norm(dense) + (1 - alpha) * norm(sparse). Both (N,) -> (N,)."""
    return alpha * min_max_normalize(dense) + (1.0 - alpha) * min_max_normalize(sparse)  # (N,)
