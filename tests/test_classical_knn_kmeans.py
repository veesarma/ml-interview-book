"""Tests for knn.py and kmeans.py — one focused test per retype target."""
import numpy as np

from mlbook.classical.kmeans import kmeans, kmeans_objective, kmeans_plusplus_init, minibatch_kmeans, squared_distances
from mlbook.classical.knn import KDTree, KNNClassifier, KNNRegressor, knn_indices, pairwise_distances


def test_pairwise_distances_match_naive():
    rng = np.random.default_rng(0)
    Q, X = rng.standard_normal((4, 3)), rng.standard_normal((6, 3))
    naive = np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.testing.assert_allclose(pairwise_distances(Q, X), naive, atol=1e-10)
    np.testing.assert_allclose(pairwise_distances(Q, X, "manhattan"), np.abs(Q[:, None] - X[None]).sum(2))
    cos = 1 - (Q @ X.T) / (np.linalg.norm(Q, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None])
    np.testing.assert_allclose(pairwise_distances(Q, X, "cosine"), cos, atol=1e-10)


def test_knn_indices_sorted_and_correct():
    rng = np.random.default_rng(0)
    X, Q = rng.standard_normal((500, 3)), rng.standard_normal((20, 3))
    idx = knn_indices(Q, X, k=5)
    D = pairwise_distances(Q, X)
    for m in range(20):
        assert np.all(np.diff(D[m, idx[m]]) >= 0)
        assert set(idx[m]) == set(np.argsort(D[m])[:5])


def test_kdtree_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    X, Q = rng.standard_normal((500, 3)), rng.standard_normal((20, 3))
    idx = knn_indices(Q, X, k=5)
    D = pairwise_distances(Q, X)
    tree = KDTree(X, leaf_size=8)
    for m in range(20):
        d_t, i_t = tree.query(Q[m], k=5)
        np.testing.assert_array_equal(i_t, idx[m])
        np.testing.assert_allclose(d_t, D[m, idx[m]])


def test_knn_classifier_blobs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = np.array([[0, 0], [4, 0], [0, 4.0]])[y] + 0.5 * rng.standard_normal((300, 2))
    assert (KNNClassifier(k=7).fit(X, y).predict(X) == y).mean() > 0.97
    assert (KNNClassifier(k=7, weighted=True).fit(X, y).predict(X) == y).mean() > 0.97


def test_knn_regressor_smooth_function():
    Xr = np.linspace(0, 1, 200)[:, None]
    yr = Xr[:, 0] ** 2
    reg = KNNRegressor(k=3).fit(Xr, yr)
    assert np.mean((reg.predict(Xr) - yr) ** 2) < 1e-3


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    C = np.array([[0, 0], [6, 0], [0, 6], [6, 6.0]])
    lab = rng.integers(0, 4, 400)
    return C[lab] + 0.7 * rng.standard_normal((400, 2)), C


def test_squared_distances_and_kmeans_objective():
    rng = np.random.default_rng(0)
    X, C = rng.standard_normal((5, 2)), rng.standard_normal((3, 2))
    naive = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(squared_distances(X, C), naive, atol=1e-10)
    labels = naive.argmin(axis=1)
    np.testing.assert_allclose(kmeans_objective(X, C, labels), naive.min(axis=1).sum())


def test_kmeans_plusplus_init_one_seed_per_blob():
    X, C_true = _blobs(1)
    seeds = kmeans_plusplus_init(X, 4, np.random.default_rng(0))
    assert seeds.shape == (4, 2)
    assert np.all(squared_distances(C_true, seeds).min(axis=1) < 4.0)


def test_kmeans_objective_monotone_and_recovers_centres():
    X, C_true = _blobs()
    C, labels, hist = kmeans(X, 4, seed=0)
    assert np.all(np.diff(hist) <= 1e-9), hist
    np.testing.assert_allclose(kmeans_objective(X, C, labels), hist[-1])
    assert np.all(squared_distances(C, C_true).min(axis=1) < 0.2)
    assert labels.shape == (400,)


def test_minibatch_kmeans_close_to_true_centres():
    X, C_true = _blobs(1)
    Cm = minibatch_kmeans(X, 4, batch_size=64, n_steps=100, seed=0)
    assert np.all(squared_distances(C_true, Cm).min(axis=1) < 1.0)
