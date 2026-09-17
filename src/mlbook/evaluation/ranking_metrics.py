"""Ranking metrics (NumPy): DCG/NDCG, MRR, MAP, precision@k, recall@k, hit rate.

Inputs follow one convention: `ranked_rels` is a (N,) array of relevance grades
in the order the system ranked the items (position 0 = top). Binary metrics treat
rel > 0 as relevant.
"""
from __future__ import annotations

import numpy as np


def dcg_at_k(ranked_rels: np.ndarray, k: int) -> float:
    """DCG@k = sum_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)."""
    r = np.asarray(ranked_rels, dtype=float)[:k]  # (k',)
    gains = 2.0**r - 1.0  # (k',)
    discounts = np.log2(np.arange(2, len(r) + 2))  # (k',) log2(i+1), i from 1
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_rels: np.ndarray, k: int) -> float:
    """NDCG@k = DCG@k / IDCG@k, IDCG from the ideal (sorted-descending) ordering."""
    ideal = np.sort(np.asarray(ranked_rels, dtype=float))[::-1]  # (N,)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(ranked_rels, k) / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked_rels: np.ndarray) -> float:
    """1 / (position of first relevant item), 0 if none."""
    hits = np.flatnonzero(np.asarray(ranked_rels) > 0)
    return 1.0 / (hits[0] + 1) if hits.size else 0.0


def precision_at_k(ranked_rels: np.ndarray, k: int) -> float:
    r = np.asarray(ranked_rels)[:k] > 0
    return float(r.sum() / k)


def recall_at_k(ranked_rels: np.ndarray, k: int, n_relevant_total: int) -> float:
    """Relevant items in the top-k over all relevant items in the corpus."""
    r = np.asarray(ranked_rels)[:k] > 0
    return float(r.sum() / max(n_relevant_total, 1))


def hit_rate_at_k(ranked_rels: np.ndarray, k: int) -> float:
    """1 if at least one relevant item in the top-k, else 0."""
    return float(np.any(np.asarray(ranked_rels)[:k] > 0))


def average_precision_ranking(ranked_rels: np.ndarray, n_relevant_total: int | None = None) -> float:
    """AP = (1/R) sum_i P@i * rel_i over positions with a relevant item."""
    r = np.asarray(ranked_rels) > 0  # (N,)
    R = n_relevant_total if n_relevant_total is not None else int(r.sum())
    if R == 0:
        return 0.0
    hits = np.cumsum(r)  # (N,)
    positions = np.arange(1, len(r) + 1)  # (N,)
    return float(np.sum((hits / positions) * r) / R)


def mean_over_queries(metric_fn, per_query: list[np.ndarray], **kwargs) -> float:
    """Average a per-query metric over a list of ranked relevance arrays."""
    return float(np.mean([metric_fn(q, **kwargs) for q in per_query]))
