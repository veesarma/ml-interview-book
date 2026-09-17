import numpy as np

from mlbook.retrieval.hybrid import convex_fusion, min_max_normalize, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_hand_computed():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    scores = dict(fused)
    assert np.isclose(scores[1], 1 / 61 + 1 / 62)
    assert np.isclose(scores[3], 1 / 63 + 1 / 61)
    assert np.isclose(scores[2], 1 / 62) and np.isclose(scores[4], 1 / 63)
    assert fused[0][0] == 1 and fused[1][0] == 3  # doc 1 tops, doc 3 second


def test_min_max_normalize_and_convex_fusion():
    s = np.array([2.0, 4.0, 6.0])
    assert np.allclose(min_max_normalize(s), [0, 0.5, 1])
    dense = np.array([0.1, 0.9, 0.5])
    sparse = np.array([10.0, 0.0, 5.0])
    f = convex_fusion(dense, sparse, alpha=0.5)
    assert np.allclose(f, [0.5, 0.5, 0.5])
