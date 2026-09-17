"""One focused test per public symbol in mlbook.math.linalg (select with -k <name>)."""

import numpy as np
import torch

from mlbook.math import linalg as la


def test_project_onto_vector():
    v = np.array([3.0, 4.0, 0.0])
    u = np.array([1.0, 0.0, 0.0])
    p = la.project_onto_vector(v, u)
    assert np.allclose(p, [3.0, 0.0, 0.0])
    assert abs((v - p) @ u) < 1e-12  # residual orthogonal to u
    u2 = np.random.randn(5)
    v2 = np.random.randn(5)
    assert abs((v2 - la.project_onto_vector(v2, u2)) @ u2) < 1e-10


def test_projection_matrix():
    A = np.random.randn(6, 2)
    P = la.projection_matrix(A)
    assert np.allclose(P @ P, P)  # idempotent
    assert np.allclose(P, P.T)  # symmetric
    assert np.allclose(P @ A, A)  # fixes the column space
    assert np.allclose(np.trace(P), 2.0)  # trace = rank


def test_least_squares_normal_equations():
    X = np.random.randn(50, 4)
    y = np.random.randn(50)
    w = la.least_squares_normal_equations(X, y)
    w_ref, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(w, w_ref)
    assert np.allclose(X.T @ (y - X @ w), 0.0, atol=1e-10)  # residual orthogonal to columns
    Y = np.random.randn(50, 3)
    assert np.allclose(la.least_squares_normal_equations(X, Y), np.linalg.lstsq(X, Y, rcond=None)[0])


def test_pseudoinverse_svd():
    A = np.random.randn(5, 3)
    assert np.allclose(la.pseudoinverse_svd(A), np.linalg.pinv(A))
    B = np.random.randn(5, 2) @ np.random.randn(2, 4)  # rank 2 in 5x4
    assert np.allclose(la.pseudoinverse_svd(B), np.linalg.pinv(B), atol=1e-8)


def test_gram_matrix():
    X = np.random.randn(7, 3)
    G = la.gram_matrix(X)
    assert G.shape == (7, 7)
    assert la.is_psd(G)
    assert np.linalg.matrix_rank(G) == 3
    assert np.isclose(G[1, 4], X[1] @ X[4])


def test_l1_norm():
    x = np.random.randn(10)
    assert np.isclose(la.l1_norm(x), np.linalg.norm(x, 1))


def test_l2_norm():
    x = np.random.randn(10)
    assert np.isclose(la.l2_norm(x), np.linalg.norm(x, 2))


def test_frobenius_norm():
    A = np.random.randn(6, 4)
    assert np.isclose(la.frobenius_norm(A), np.linalg.norm(A, "fro"))
    s = np.linalg.svd(A, compute_uv=False)
    assert np.isclose(la.frobenius_norm(A), np.sqrt(np.sum(s**2)))


def test_spectral_norm_power_iteration():
    A = np.random.randn(6, 4)
    assert np.isclose(la.spectral_norm_power_iteration(A, n_iter=500), np.linalg.norm(A, 2), rtol=1e-6)
    x = np.random.randn(4)
    assert np.linalg.norm(A @ x) <= la.spectral_norm_power_iteration(A) * np.linalg.norm(x) + 1e-9


def test_power_iteration():
    M = np.random.randn(5, 5)
    A = M @ M.T + np.eye(5)
    lam, v = la.power_iteration(A, n_iter=500)
    assert np.isclose(lam, np.linalg.eigvalsh(A)[-1], rtol=1e-6)
    assert np.allclose(A @ v, lam * v, atol=1e-5)


def test_is_psd():
    M = np.random.randn(4, 4)
    assert la.is_psd(M @ M.T)
    assert not la.is_psd(np.diag([1.0, -1.0]))
    assert not la.is_psd(np.array([[1.0, 2.0], [0.0, 1.0]]))  # not symmetric


def test_quadratic_form():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    x = np.array([1.0, -1.0])
    assert np.isclose(la.quadratic_form(A, x), 2 - 2 + 3)


def test_low_rank_approx():
    A = np.random.randn(20, 12)
    k = 3
    A_k = la.low_rank_approx(A, k)
    s = np.linalg.svd(A, compute_uv=False)
    assert np.linalg.matrix_rank(A_k) == k
    assert np.isclose(la.frobenius_norm(A - A_k), np.sqrt(np.sum(s[k:] ** 2)))  # Eckart-Young
    assert np.isclose(np.linalg.norm(A - A_k, 2), s[k])
    R = np.random.randn(20, k) @ np.random.randn(k, 12)
    assert la.frobenius_norm(A - R) >= la.frobenius_norm(A - A_k)


def _pca_data():
    return np.random.randn(200, 5) @ np.diag([5, 3, 1, 0.5, 0.1]) + 2.0


def test_pca_eig():
    X = _pca_data()
    W, var = la.pca_eig(X, 2)
    cov = np.cov(X, rowvar=False)
    assert np.allclose(var, np.sort(np.linalg.eigvalsh(cov))[::-1][:2])
    assert np.allclose(W.T @ W, np.eye(2), atol=1e-10)  # orthonormal columns
    assert np.allclose(cov @ W[:, 0], var[0] * W[:, 0])  # eigenvector equation


def test_pca_svd():
    X = _pca_data()
    W_eig, var_eig = la.pca_eig(X, 2)
    W_svd, var_svd = la.pca_svd(X, 2)
    assert np.allclose(var_eig, var_svd)
    for j in range(2):  # eigenvectors unique up to sign
        assert np.isclose(abs(W_eig[:, j] @ W_svd[:, j]), 1.0)


def test_softmax_rows():
    S = np.random.randn(4, 6) * 50  # large logits: must not overflow
    A = la.softmax_rows(S)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert np.allclose(A, torch.softmax(torch.tensor(S), dim=-1).numpy())


def test_attention_numpy():
    T, d_k, d_v = 4, 8, 6
    Q, K, V = (np.random.randn(T, d) for d in (d_k, d_k, d_v))
    Y, A = la.attention_numpy(Q, K, V)
    assert np.allclose(A.sum(axis=1), 1.0)
    Y_ref = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(Q)[None], torch.tensor(K)[None], torch.tensor(V)[None]
    )[0].numpy()
    assert np.allclose(Y, Y_ref, atol=1e-10)


def test_kronecker():
    A = np.random.randn(2, 3)
    B = np.random.randn(4, 2)
    assert np.allclose(la.kronecker(A, B), np.kron(A, B))
