import numpy as np

from mlbook.retrieval.hnsw import HNSW
from mlbook.retrieval.similarity import brute_force_topk


def test_hnsw_search_layer_finds_exact_neighbours_on_small_graph():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    idx = HNSW(M=6, ef_construction=40)
    for x in X:
        idx.add(x)
    q = rng.normal(size=4)
    found = idx._search_layer(q, idx.entry if idx.levels[idx.entry] == 0 else 0, ef=60, layer=0)
    ids = np.array([i for _, i in found[:5]])
    truth, _ = brute_force_topk(q[None, :], X, 5, "l2")
    assert np.array_equal(ids, truth[0])  # ef = N explores everything reachable


def test_hnsw_levels_geometric_and_graph_degrees_bounded():
    rng = np.random.default_rng(0)
    idx = HNSW(M=4, ef_construction=20)
    for x in rng.normal(size=(300, 8)):
        idx.add(x)
    levels = np.array(idx.levels)
    assert (levels == 0).mean() > 0.5 and levels.max() >= 1
    assert all(len(nbrs) <= idx.M_max0 for nbrs in idx.graph[0].values())
    for l in range(1, len(idx.graph)):
        assert all(len(nbrs) <= idx.M for nbrs in idx.graph[l].values())


def test_hnsw_recall_at_10_high_and_improves_with_ef():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 8))
    idx = HNSW(M=8, ef_construction=64)
    for x in X:
        idx.add(x)
    Q = rng.normal(size=(20, 8))
    truth, _ = brute_force_topk(Q, X, 10, "l2")
    rec = {}
    for ef in (10, 100):
        hits = 0
        for i in range(20):
            ids, d2 = idx.search(Q[i], 10, ef=ef)
            assert np.all(np.diff(d2) >= 0)
            hits += len(set(ids) & set(truth[i]))
        rec[ef] = hits / 200
    assert rec[100] >= rec[10] and rec[100] >= 0.9
