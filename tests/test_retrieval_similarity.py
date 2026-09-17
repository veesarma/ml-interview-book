import numpy as np

from mlbook.retrieval import similarity as sim


def test_normalize_rows_unit_norm():
    X = np.random.randn(5, 4)
    assert np.allclose(np.linalg.norm(sim.normalize_rows(X), axis=1), 1.0)


def test_dot_product_matches_numpy():
    Q, X = np.random.randn(3, 4), np.random.randn(6, 4)
    assert np.allclose(sim.dot_product(Q, X), Q @ X.T)


def test_cosine_similarity_range_and_self():
    X = np.random.randn(6, 4)
    S = sim.cosine_similarity(X, X)
    assert np.allclose(np.diag(S), 1.0) and S.max() <= 1 + 1e-9 and S.min() >= -1 - 1e-9


def test_squared_euclidean_matches_direct():
    Q, X = np.random.randn(3, 4), np.random.randn(6, 4)
    direct = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    assert np.allclose(sim.squared_euclidean(Q, X), direct)


def test_normalised_vectors_make_metrics_agree():
    Q = sim.normalize_rows(np.random.randn(3, 8))
    X = sim.normalize_rows(np.random.randn(20, 8))
    assert np.allclose(sim.squared_euclidean(Q, X), 2 - 2 * sim.cosine_similarity(Q, X), atol=1e-9)
    ids_dot, _ = sim.brute_force_topk(Q, X, 5, "dot")
    ids_l2, _ = sim.brute_force_topk(Q, X, 5, "l2")
    assert np.array_equal(ids_dot, ids_l2)


def test_mips_to_nn_reduction_recovers_argmax_inner_product():
    X = np.random.randn(50, 6) * np.random.uniform(0.5, 3.0, size=(50, 1))  # varied norms
    Q = np.random.randn(4, 6)
    Xa, Qa = sim.mips_to_nn_corpus(X), sim.mips_to_nn_query(Q)
    assert np.allclose(np.linalg.norm(Xa, axis=1), np.linalg.norm(Xa, axis=1)[0])
    ids_mips = np.argmax(Q @ X.T, axis=1)
    ids_l2, _ = sim.brute_force_topk(Qa, Xa, 1, "l2")
    assert np.array_equal(ids_mips, ids_l2[:, 0])


def test_brute_force_topk_ordering():
    Q, X = np.random.randn(2, 4), np.random.randn(30, 4)
    ids, scores = sim.brute_force_topk(Q, X, 5, "dot")
    S = Q @ X.T
    assert np.array_equal(ids, np.argsort(-S, axis=1)[:, :5])
    assert np.all(np.diff(scores, axis=1) <= 0)
