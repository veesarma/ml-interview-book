import numpy as np
import torch

from mlbook.math import linalg as la


def test_projection_onto_vector_is_orthogonal_residual():
    v = np.array([3.0, 4.0, 0.0])
    u = np.array([1.0, 0.0, 0.0])
    p = la.project_onto_vector(v, u)
    assert np.allclose(p, [3.0, 0.0, 0.0])
    assert abs((v - p) @ u) < 1e-12  # residual orthogonal to u


def test_projection_matrix_is_idempotent_and_symmetric():
    A = np.random.randn(6, 2)
    P = la.projection_matrix(A)
    assert np.allclose(P @ P, P)
    assert np.allclose(P, P.T)
    assert np.allclose(P @ A, A)  # column space is fixed


def test_normal_equations_match_lstsq():
    X = np.random.randn(50, 4)
    y = np.random.randn(50)
    w = la.least_squares_normal_equations(X, y)
    w_ref, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(w, w_ref)
    # residual orthogonal to every column: X^T (y - Xw) = 0
    assert np.allclose(X.T @ (y - X @ w), 0.0, atol=1e-10)


def test_pseudoinverse_matches_numpy_including_rank_deficient():
    A = np.random.randn(5, 3)
    assert np.allclose(la.pseudoinverse_svd(A), np.linalg.pinv(A))
    B = np.random.randn(5, 2) @ np.random.randn(2, 4)  # rank 2 in 5x4
    assert np.allclose(la.pseudoinverse_svd(B), np.linalg.pinv(B), atol=1e-8)


def test_gram_matrix_is_psd():
    X = np.random.randn(7, 3)
    G = la.gram_matrix(X)
    assert la.is_psd(G)
    assert np.linalg.matrix_rank(G) == 3


def test_norms_against_numpy():
    x = np.random.randn(10)
    A = np.random.randn(6, 4)
    assert np.isclose(la.l1_norm(x), np.linalg.norm(x, 1))
    assert np.isclose(la.l2_norm(x), np.linalg.norm(x, 2))
    assert np.isclose(la.frobenius_norm(A), np.linalg.norm(A, "fro"))
    assert np.isclose(la.spectral_norm_power_iteration(A, n_iter=500), np.linalg.norm(A, 2), rtol=1e-6)


def test_power_iteration_finds_top_eigenpair():
    M = np.random.randn(5, 5)
    A = M @ M.T + np.eye(5)
    lam, v = la.power_iteration(A, n_iter=500)
    lam_ref = np.linalg.eigvalsh(A)[-1]
    assert np.isclose(lam, lam_ref, rtol=1e-6)
    assert np.allclose(A @ v, lam * v, atol=1e-5)


def test_low_rank_approx_eckart_young():
    A = np.random.randn(20, 12)
    k = 3
    A_k = la.low_rank_approx(A, k)
    s = np.linalg.svd(A, compute_uv=False)
    assert np.linalg.matrix_rank(A_k) == k
    # Frobenius error equals sqrt of the discarded singular values squared.
    assert np.isclose(la.frobenius_norm(A - A_k), np.sqrt(np.sum(s[k:] ** 2)))
    # A random rank-k matrix cannot beat it.
    R = np.random.randn(20, k) @ np.random.randn(k, 12)
    assert la.frobenius_norm(A - R) >= la.frobenius_norm(A - A_k)


def test_pca_eig_and_svd_agree():
    X = np.random.randn(200, 5) @ np.diag([5, 3, 1, 0.5, 0.1]) + 2.0
    W_eig, var_eig = la.pca_eig(X, 2)
    W_svd, var_svd = la.pca_svd(X, 2)
    assert np.allclose(var_eig, var_svd)
    # eigenvectors are unique up to sign
    for j in range(2):
        assert np.isclose(abs(W_eig[:, j] @ W_svd[:, j]), 1.0)
    # equals explained variance from covariance eigvals
    cov = np.cov(X, rowvar=False)
    assert np.allclose(var_eig, np.sort(np.linalg.eigvalsh(cov))[::-1][:2])


def test_attention_matches_torch():
    T, d_k, d_v = 4, 8, 6
    Q, K, V = (np.random.randn(T, d) for d in (d_k, d_k, d_v))
    Y, A = la.attention_numpy(Q, K, V)
    assert np.allclose(A.sum(axis=1), 1.0)
    Y_ref = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(Q)[None], torch.tensor(K)[None], torch.tensor(V)[None]
    )[0].numpy()
    assert np.allclose(Y, Y_ref, atol=1e-10)


def test_kronecker_matches_numpy():
    A = np.random.randn(2, 3)
    B = np.random.randn(4, 2)
    assert np.allclose(la.kronecker(A, B), np.kron(A, B))
