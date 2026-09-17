import numpy as np

from mlbook.evaluation import ranking_metrics as rm


def test_dcg_and_ndcg_hand_computed():
    rels = np.array([3, 2, 3, 0, 1, 2])
    dcg = (2**3 - 1) / 1 + (2**2 - 1) / np.log2(3) + 7 / 2 + 0 + 1 / np.log2(6) + 3 / np.log2(7)
    assert np.isclose(rm.dcg_at_k(rels, 6), dcg)
    ideal = np.array([3, 3, 2, 2, 1, 0])
    assert np.isclose(rm.ndcg_at_k(rels, 6), dcg / rm.dcg_at_k(ideal, 6))
    assert rm.ndcg_at_k(ideal, 6) == 1.0


def test_reciprocal_rank():
    assert rm.reciprocal_rank(np.array([0, 0, 1, 1])) == 1 / 3
    assert rm.reciprocal_rank(np.array([0, 0])) == 0.0


def test_precision_recall_hit_at_k():
    r = np.array([1, 0, 1, 0, 0])
    assert rm.precision_at_k(r, 3) == 2 / 3
    assert rm.recall_at_k(r, 3, n_relevant_total=4) == 0.5
    assert rm.hit_rate_at_k(r, 1) == 1.0 and rm.hit_rate_at_k(np.array([0, 0, 1]), 2) == 0.0


def test_average_precision_ranking_hand_computed():
    r = np.array([1, 0, 1, 0, 1])
    assert np.isclose(rm.average_precision_ranking(r), (1 + 2 / 3 + 3 / 5) / 3)
    assert np.isclose(rm.average_precision_ranking(r, n_relevant_total=5), (1 + 2 / 3 + 3 / 5) / 5)


def test_mean_over_queries():
    qs = [np.array([1, 0]), np.array([0, 1])]
    assert np.isclose(rm.mean_over_queries(rm.reciprocal_rank, qs), 0.75)
