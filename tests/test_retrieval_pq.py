import numpy as np

from mlbook.retrieval.pq import ProductQuantizer
from mlbook.retrieval.similarity import brute_force_topk


def test_pq_encode_decode_shapes_and_reconstruction_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    pq = ProductQuantizer(n_subvectors=4, n_codes=16, n_iters=10).train(X)
    codes = pq.encode(X)
    assert codes.shape == (400, 4) and codes.max() < 16
    Xr = pq.decode(codes)
    err = np.mean((X - Xr) ** 2)
    assert err < np.mean(X**2) * 0.8  # quantisation beats predicting zero
    assert pq.bytes_per_vector() == 4 * 4 / 8  # 4 codes x 4 bits


def test_pq_asymmetric_distance_equals_distance_to_reconstruction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    pq = ProductQuantizer(n_subvectors=2, n_codes=8, n_iters=10).train(X)
    codes = pq.encode(X)
    q = rng.normal(size=8)
    adc = pq.asymmetric_distances(q, codes)
    exact_to_recon = ((pq.decode(codes) - q) ** 2).sum(axis=1)
    assert np.allclose(adc, exact_to_recon)
    table = pq.distance_table(q)
    assert table.shape == (2, 8)


def test_pq_search_recall_reasonable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 16))
    pq = ProductQuantizer(n_subvectors=8, n_codes=32, n_iters=10).train(X)
    codes = pq.encode(X)
    Q = rng.normal(size=(20, 16))
    truth, _ = brute_force_topk(Q, X, 10, "l2")
    hits = sum(len(set(pq.search(Q[i], codes, 10)[0]) & set(truth[i])) for i in range(20))
    assert hits / 200 > 0.4
