# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/evaluation/ranking_metrics.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ranking_metrics -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py evaluation/ranking_metrics --force

"""Ranking metrics (NumPy): DCG/NDCG, MRR, MAP, precision@k, recall@k, hit rate.

Inputs follow one convention: `ranked_rels` is a (N,) array of relevance grades
in the order the system ranked the items (position 0 = top). Binary metrics treat
rel > 0 as relevant.
"""
from __future__ import annotations
import numpy as np

def dcg_at_k(ranked_rels: np.ndarray, k: int) -> float:
    """DCG@k = sum_{i=1..k} (2^{rel_i} - 1) / log2(i + 1)."""
    raise NotImplementedError('TODO: implement dcg_at_k (see the reference in src/mlbook)')

def ndcg_at_k(ranked_rels: np.ndarray, k: int) -> float:
    """NDCG@k = DCG@k / IDCG@k, IDCG from the ideal (sorted-descending) ordering."""
    raise NotImplementedError('TODO: implement ndcg_at_k (see the reference in src/mlbook)')

def reciprocal_rank(ranked_rels: np.ndarray) -> float:
    """1 / (position of first relevant item), 0 if none."""
    raise NotImplementedError('TODO: implement reciprocal_rank (see the reference in src/mlbook)')

def precision_at_k(ranked_rels: np.ndarray, k: int) -> float:
    raise NotImplementedError('TODO: implement precision_at_k (see the reference in src/mlbook)')

def recall_at_k(ranked_rels: np.ndarray, k: int, n_relevant_total: int) -> float:
    """Relevant items in the top-k over all relevant items in the corpus."""
    raise NotImplementedError('TODO: implement recall_at_k (see the reference in src/mlbook)')

def hit_rate_at_k(ranked_rels: np.ndarray, k: int) -> float:
    """1 if at least one relevant item in the top-k, else 0."""
    raise NotImplementedError('TODO: implement hit_rate_at_k (see the reference in src/mlbook)')

def average_precision_ranking(ranked_rels: np.ndarray, n_relevant_total: int | None=None) -> float:
    """AP = (1/R) sum_i P@i * rel_i over positions with a relevant item."""
    raise NotImplementedError('TODO: implement average_precision_ranking (see the reference in src/mlbook)')

def mean_over_queries(metric_fn, per_query: list[np.ndarray], **kwargs) -> float:
    """Average a per-query metric over a list of ranked relevance arrays."""
    raise NotImplementedError('TODO: implement mean_over_queries (see the reference in src/mlbook)')
