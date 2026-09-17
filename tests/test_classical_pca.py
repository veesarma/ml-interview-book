"""Tests for pca.py — one focused test per retype target."""
import numpy as np

from mlbook.classical import pca


def _data(N=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((N, 2)) * np.array([3.0, 1.0])  # (N, 2)
    mixing = rng.standard_normal((2, d))  # (2, d)
    return latent @ mixing + 0.1 * rng.standard_normal((N, d)) + 5.0


def test_pca_eig_matches_numpy_svd():
    X = _data()
    V, var, mu = pca.pca_eig(X, 2)
    _, s, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    np.testing.assert_allclose(np.abs(V.T), np.abs(Vt[:2]), atol=1e-8)
    np.testing.assert_allclose(var, s[:2] ** 2 / (X.shape[0] - 1))
    np.testing.assert_allclose(mu, X.mean(0))


def test_pca_svd_matches_numpy_svd_and_pca_eig():
    X = _data()
    V_s, var_s, mu = pca.pca_svd(X, 2)
    V_e, var_e, _ = pca.pca_eig(X, 2)
    np.testing.assert_allclose(np.abs(V_e), np.abs(V_s), atol=1e-8)
    np.testing.assert_allclose(var_e, var_s, atol=1e-8)
    _, s, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    np.testing.assert_allclose(np.abs(V_s.T), np.abs(Vt[:2]), atol=1e-8)
    Z = pca.transform(X, V_s, mu)
    assert Z.shape == (300, 2)
    np.testing.assert_allclose(np.abs(Z), np.abs((X - X.mean(0)) @ Vt[:2].T), atol=1e-8)


def test_transform_inverse_transform_reconstruction_error_equals_dropped_variance():
    X = _data()
    N = X.shape[0]
    V, var, mu = pca.pca_svd(X, 2)
    X_hat = pca.inverse_transform(pca.transform(X, V, mu), V, mu)
    total = np.trace(np.cov(X.T))
    np.testing.assert_allclose(((X - X_hat) ** 2).sum() / (N - 1), total - var.sum(), atol=1e-8)
    assert pca.explained_variance_ratio(var, total).sum() > 0.99


def test_whiten_gives_identity_covariance():
    X = _data()
    V, var, mu = pca.pca_svd(X, 2)
    Zw = pca.whiten(X, V, var, mu)
    np.testing.assert_allclose(np.cov(Zw.T), np.eye(2), atol=1e-6)


def test_randomized_svd_close_to_exact():
    rng = np.random.default_rng(0)
    U0, _ = np.linalg.qr(rng.standard_normal((400, 50)))  # (400, 50) orthonormal columns
    V0, _ = np.linalg.qr(rng.standard_normal((50, 50)))  # (50, 50) orthogonal
    spectrum = np.concatenate([np.linspace(10, 6, 5), np.linspace(0.5, 0.05, 45)])  # gap after rank 5
    A = U0 @ np.diag(spectrum) @ V0.T  # (400, 50)
    U, s, Vt = pca.randomized_svd(A, rank=5, n_power_iters=3)
    np.testing.assert_allclose(s, spectrum[:5], rtol=1e-3)
    A5 = U @ np.diag(s) @ Vt
    A5_ref = U0[:, :5] @ np.diag(spectrum[:5]) @ V0[:, :5].T
    assert np.linalg.norm(A5 - A5_ref) / np.linalg.norm(A5_ref) < 1e-2
