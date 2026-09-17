import numpy as np

from mlbook.retrieval.ivf import IVFIndex, kmeans, kmeans_pp_init
from mlbook.retrieval.similarity import brute_force_topk


def _blobs(n_per=100, k=5, d=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(k, d) * 15.0  # (k, d) well-separated blob centres
    X = np.concatenate([c + rng.normal(size=(n_per, d)) for c in centers])
    return X


def test_kmeans_pp_init_picks_one_centre_per_blob():
    X = _blobs()
    C = kmeans_pp_init(X, 5, np.random.default_rng(1))
    assert C.shape == (5, 8)
    assert all(np.any(np.all(X == c, axis=1)) for c in C)  # every centre is a data row
    blob_of = np.argmin(((C[:, None, :] - X[None, ::100, :]) ** 2).sum(-1), axis=1)
    assert len(np.unique(blob_of)) == 5


def test_kmeans_recovers_well_separated_clusters():
    X = _blobs()
    cents, assign = kmeans(X, 5, n_iters=30)
    assert cents.shape == (5, 8) and assign.shape == (500,)
    # each true blob should map to a single learned centroid
    for b in range(5):
        assert len(np.unique(assign[b * 100 : (b + 1) * 100])) == 1


def test_ivf_index_nprobe_all_lists_equals_brute_force():
    X = _blobs()
    idx = IVFIndex(n_list=5).train(X).add(X)
    assert sum(len(l) for l in idx.lists) == 500
    q = X[7] + 0.01
    ids_ivf, d2 = idx.search(q, 10, nprobe=5)
    ids_bf, d2_bf = brute_force_topk(q[None, :], X, 10, "l2")
    assert np.array_equal(ids_ivf, ids_bf[0]) and np.allclose(d2, d2_bf[0])


def test_ivf_recall_increases_with_nprobe():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 16))
    idx = IVFIndex(n_list=20).train(X).add(X)
    Q = rng.normal(size=(30, 16))
    truth, _ = brute_force_topk(Q, X, 10, "l2")
    recalls = []
    for nprobe in (1, 4, 20):
        hits = 0
        for i in range(30):
            ids, _ = idx.search(Q[i], 10, nprobe=nprobe)
            hits += len(set(ids) & set(truth[i]))
        recalls.append(hits / 300)
    assert recalls[0] <= recalls[1] <= recalls[2] == 1.0
