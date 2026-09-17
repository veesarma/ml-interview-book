# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/math/linalg.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k linalg -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py math/linalg --force

"""Linear algebra primitives used throughout the book (pure NumPy).

Conventions
-----------
* Row-major data: ``X`` has shape ``(N, d)`` with one example per row, so a
  linear map is ``X @ W`` with ``W`` of shape ``(d, k)``.
* Vectors are 1-D arrays of shape ``(d,)``.
* Every function documents input shapes, output shapes, and the equation.
"""
from __future__ import annotations
import numpy as np

def project_onto_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Orthogonal projection of ``v`` onto the line spanned by ``u``.

    Equation: ``proj_u(v) = (u^T v / u^T u) u``.

    Args:
        v: (d,) vector to project.
        u: (d,) direction (need not be unit length).
    Returns:
        (d,) the component of ``v`` along ``u``.
    """
    raise NotImplementedError('TODO: implement project_onto_vector (see the reference in src/mlbook)')

def projection_matrix(A: np.ndarray) -> np.ndarray:
    """Projection onto the column space of ``A``: ``P = A (A^T A)^{-1} A^T``.

    Args:
        A: (d, k) matrix with linearly independent columns (k <= d).
    Returns:
        (d, d) symmetric idempotent matrix (P @ P == P, P.T == P).
    """
    raise NotImplementedError('TODO: implement projection_matrix (see the reference in src/mlbook)')

def least_squares_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``min_w ||X w - y||^2`` via the normal equations ``X^T X w = X^T y``.

    Args:
        X: (N, d) design matrix, full column rank.
        y: (N,) or (N, k) targets.
    Returns:
        (d,) or (d, k) least-squares weights.
    """
    raise NotImplementedError('TODO: implement least_squares_normal_equations (see the reference in src/mlbook)')

def pseudoinverse_svd(A: np.ndarray, rcond: float=1e-12) -> np.ndarray:
    """Moore-Penrose pseudoinverse via SVD: ``A^+ = V S^+ U^T``.

    Singular values below ``rcond * s_max`` are treated as zero, which is what
    makes the pseudoinverse well-defined for rank-deficient ``A``.

    Args:
        A: (m, n) any matrix.
    Returns:
        (n, m) pseudoinverse.
    """
    raise NotImplementedError('TODO: implement pseudoinverse_svd (see the reference in src/mlbook)')

def gram_matrix(X: np.ndarray) -> np.ndarray:
    """Gram matrix of the rows of ``X``: ``G = X X^T``, ``G_ij = <x_i, x_j>``.

    Args:
        X: (N, d) rows are vectors.
    Returns:
        (N, N) symmetric PSD matrix of pairwise inner products.
    """
    raise NotImplementedError('TODO: implement gram_matrix (see the reference in src/mlbook)')

def l1_norm(x: np.ndarray) -> float:
    """``||x||_1 = sum |x_i|`` for a vector of shape (d,)."""
    raise NotImplementedError('TODO: implement l1_norm (see the reference in src/mlbook)')

def l2_norm(x: np.ndarray) -> float:
    """``||x||_2 = sqrt(sum x_i^2)`` for a vector of shape (d,)."""
    raise NotImplementedError('TODO: implement l2_norm (see the reference in src/mlbook)')

def frobenius_norm(A: np.ndarray) -> float:
    """``||A||_F = sqrt(sum_ij A_ij^2) = sqrt(tr(A^T A))`` for a matrix (m, n)."""
    raise NotImplementedError('TODO: implement frobenius_norm (see the reference in src/mlbook)')

def spectral_norm_power_iteration(A: np.ndarray, n_iter: int=100, seed: int=0) -> float:
    """Largest singular value ``||A||_2 = sigma_max(A)`` by power iteration on ``A^T A``.

    Iterates ``v <- A^T A v / ||A^T A v||``; then ``sigma_max = ||A v||``.
    This is the estimator used by spectral normalisation in GANs.

    Args:
        A: (m, n) matrix.
        n_iter: number of power-iteration steps.
    Returns:
        scalar estimate of the spectral norm.
    """
    raise NotImplementedError('TODO: implement spectral_norm_power_iteration (see the reference in src/mlbook)')

def power_iteration(A: np.ndarray, n_iter: int=200, seed: int=0) -> tuple[float, np.ndarray]:
    """Dominant eigenpair of a symmetric matrix by power iteration.

    Iterates ``v <- A v / ||A v||``; the Rayleigh quotient ``v^T A v`` is the
    eigenvalue estimate. Converges at rate ``|lambda_2 / lambda_1|^t``.

    Args:
        A: (d, d) symmetric matrix.
    Returns:
        (eigenvalue, eigenvector of shape (d,) with unit norm).
    """
    raise NotImplementedError('TODO: implement power_iteration (see the reference in src/mlbook)')

def is_psd(A: np.ndarray, tol: float=1e-10) -> bool:
    """True if ``A`` is symmetric and all eigenvalues are >= -tol.

    Args:
        A: (d, d) matrix.
    """
    raise NotImplementedError('TODO: implement is_psd (see the reference in src/mlbook)')

def quadratic_form(A: np.ndarray, x: np.ndarray) -> float:
    """``x^T A x`` for ``A`` (d, d) and ``x`` (d,)."""
    raise NotImplementedError('TODO: implement quadratic_form (see the reference in src/mlbook)')

def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-``k`` approximation in Frobenius/spectral norm (Eckart-Young).

    ``A_k = U_k diag(s_k) V_k^T`` using the top-k singular triplets.

    Args:
        A: (m, n) matrix.
        k: target rank, 1 <= k <= min(m, n).
    Returns:
        (m, n) rank-k matrix.
    """
    raise NotImplementedError('TODO: implement low_rank_approx (see the reference in src/mlbook)')

def pca_eig(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA via the eigen-decomposition of the covariance matrix.

    Solves ``max_{||w||=1} w^T Sigma w`` whose stationarity condition is the
    eigenproblem ``Sigma w = lambda w``.

    Args:
        X: (N, d) data, one example per row (centred inside).
        k: number of components.
    Returns:
        components: (d, k) top-k eigenvectors of Sigma (columns, unit norm).
        explained_variance: (k,) eigenvalues, descending.
    """
    raise NotImplementedError('TODO: implement pca_eig (see the reference in src/mlbook)')

def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA via SVD of the centred data matrix ``X_c = U S V^T``.

    ``Sigma = X_c^T X_c / (N-1) = V (S^2 / (N-1)) V^T``: the right singular
    vectors are the principal directions and ``s^2/(N-1)`` the variances.

    Args:
        X: (N, d) data.
        k: number of components.
    Returns:
        components: (d, k) right singular vectors.
        explained_variance: (k,) ``s_i^2 / (N - 1)``.
    """
    raise NotImplementedError('TODO: implement pca_svd (see the reference in src/mlbook)')

def softmax_rows(S: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the max-subtraction trick.

    Args:
        S: (T, T) scores (or any (..., n) array; softmax is over the last axis).
    Returns:
        (T, T) rows sum to one.
    """
    raise NotImplementedError('TODO: implement softmax_rows (see the reference in src/mlbook)')

def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-head scaled dot-product attention as three matrix operations.

    ``S = Q K^T / sqrt(d_k)`` (Gram-like similarity), ``A = softmax_rows(S)``
    (row-stochastic), ``Y = A V`` (each output row is a convex combination of
    value rows).

    Args:
        Q: (T, d_k) queries. K: (T, d_k) keys. V: (T, d_v) values.
    Returns:
        Y: (T, d_v) outputs; A: (T, T) attention weights.
    """
    raise NotImplementedError('TODO: implement attention_numpy (see the reference in src/mlbook)')

def kronecker(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product ``A (x) B`` written out explicitly.

    ``(A (x) B)[i*p + k, j*q + l] = A[i, j] * B[k, l]``.

    Args:
        A: (m, n). B: (p, q).
    Returns:
        (m*p, n*q).
    """
    raise NotImplementedError('TODO: implement kronecker (see the reference in src/mlbook)')
