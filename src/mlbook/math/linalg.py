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

# ---------------------------------------------------------------------------
# Projections and least squares
# ---------------------------------------------------------------------------


def project_onto_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Orthogonal projection of ``v`` onto the line spanned by ``u``.

    Equation: ``proj_u(v) = (u^T v / u^T u) u``.

    Args:
        v: (d,) vector to project.
        u: (d,) direction (need not be unit length).
    Returns:
        (d,) the component of ``v`` along ``u``.
    """
    coeff = (u @ v) / (u @ u)  # scalar: <u, v> / <u, u>
    return coeff * u  # (d,)


def projection_matrix(A: np.ndarray) -> np.ndarray:
    """Projection onto the column space of ``A``: ``P = A (A^T A)^{-1} A^T``.

    Args:
        A: (d, k) matrix with linearly independent columns (k <= d).
    Returns:
        (d, d) symmetric idempotent matrix (P @ P == P, P.T == P).
    """
    gram = A.T @ A  # (k, k) Gram matrix of the columns
    P = A @ np.linalg.solve(gram, A.T)  # (d, k) @ (k, d) -> (d, d)
    return P


def least_squares_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``min_w ||X w - y||^2`` via the normal equations ``X^T X w = X^T y``.

    Args:
        X: (N, d) design matrix, full column rank.
        y: (N,) or (N, k) targets.
    Returns:
        (d,) or (d, k) least-squares weights.
    """
    XtX = X.T @ X  # (d, d)
    Xty = X.T @ y  # (d,) or (d, k)
    return np.linalg.solve(XtX, Xty)  # (d,) or (d, k)


def pseudoinverse_svd(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Moore-Penrose pseudoinverse via SVD: ``A^+ = V S^+ U^T``.

    Singular values below ``rcond * s_max`` are treated as zero, which is what
    makes the pseudoinverse well-defined for rank-deficient ``A``.

    Args:
        A: (m, n) any matrix.
    Returns:
        (n, m) pseudoinverse.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # (m, r), (r,), (r, n)
    cutoff = rcond * s.max()
    s_inv = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)  # (r,)
    return Vt.T @ (s_inv[:, None] * U.T)  # (n, r) @ (r, m) -> (n, m)


def gram_matrix(X: np.ndarray) -> np.ndarray:
    """Gram matrix of the rows of ``X``: ``G = X X^T``, ``G_ij = <x_i, x_j>``.

    Args:
        X: (N, d) rows are vectors.
    Returns:
        (N, N) symmetric PSD matrix of pairwise inner products.
    """
    return X @ X.T  # (N, N)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


def l1_norm(x: np.ndarray) -> float:
    """``||x||_1 = sum |x_i|`` for a vector of shape (d,)."""
    return float(np.sum(np.abs(x)))


def l2_norm(x: np.ndarray) -> float:
    """``||x||_2 = sqrt(sum x_i^2)`` for a vector of shape (d,)."""
    return float(np.sqrt(np.sum(x * x)))


def frobenius_norm(A: np.ndarray) -> float:
    """``||A||_F = sqrt(sum_ij A_ij^2) = sqrt(tr(A^T A))`` for a matrix (m, n)."""
    return float(np.sqrt(np.sum(A * A)))


def spectral_norm_power_iteration(
    A: np.ndarray, n_iter: int = 100, seed: int = 0
) -> float:
    """Largest singular value ``||A||_2 = sigma_max(A)`` by power iteration on ``A^T A``.

    Iterates ``v <- A^T A v / ||A^T A v||``; then ``sigma_max = ||A v||``.
    This is the estimator used by spectral normalisation in GANs.

    Args:
        A: (m, n) matrix.
        n_iter: number of power-iteration steps.
    Returns:
        scalar estimate of the spectral norm.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])  # (n,)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        u = A @ v  # (m,)
        v = A.T @ u  # (n,)
        v /= np.linalg.norm(v) + 1e-12
    return float(np.linalg.norm(A @ v))  # ||A v|| with ||v|| = 1


# ---------------------------------------------------------------------------
# Eigen-decomposition, PSD, SVD, low rank
# ---------------------------------------------------------------------------


def power_iteration(A: np.ndarray, n_iter: int = 200, seed: int = 0) -> tuple[float, np.ndarray]:
    """Dominant eigenpair of a symmetric matrix by power iteration.

    Iterates ``v <- A v / ||A v||``; the Rayleigh quotient ``v^T A v`` is the
    eigenvalue estimate. Converges at rate ``|lambda_2 / lambda_1|^t``.

    Args:
        A: (d, d) symmetric matrix.
    Returns:
        (eigenvalue, eigenvector of shape (d,) with unit norm).
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])  # (d,)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        Av = A @ v  # (d,)
        v = Av / (np.linalg.norm(Av) + 1e-12)
    lam = float(v @ A @ v)  # Rayleigh quotient
    return lam, v


def is_psd(A: np.ndarray, tol: float = 1e-10) -> bool:
    """True if ``A`` is symmetric and all eigenvalues are >= -tol.

    Args:
        A: (d, d) matrix.
    """
    if not np.allclose(A, A.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(A)  # (d,) ascending
    return bool(eigvals.min() >= -tol)


def quadratic_form(A: np.ndarray, x: np.ndarray) -> float:
    """``x^T A x`` for ``A`` (d, d) and ``x`` (d,)."""
    return float(x @ A @ x)


def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-``k`` approximation in Frobenius/spectral norm (Eckart-Young).

    ``A_k = U_k diag(s_k) V_k^T`` using the top-k singular triplets.

    Args:
        A: (m, n) matrix.
        k: target rank, 1 <= k <= min(m, n).
    Returns:
        (m, n) rank-k matrix.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # (m, r), (r,), (r, n)
    Uk = U[:, :k]  # (m, k)
    sk = s[:k]  # (k,)
    Vtk = Vt[:k, :]  # (k, n)
    return (Uk * sk) @ Vtk  # (m, k) @ (k, n) -> (m, n)


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
    Xc = X - X.mean(axis=0, keepdims=True)  # (N, d)
    Sigma = Xc.T @ Xc / (X.shape[0] - 1)  # (d, d) sample covariance
    eigvals, eigvecs = np.linalg.eigh(Sigma)  # (d,), (d, d) ascending
    order = np.argsort(eigvals)[::-1]  # (d,) descending
    return eigvecs[:, order[:k]], eigvals[order[:k]]  # (d, k), (k,)


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
    Xc = X - X.mean(axis=0, keepdims=True)  # (N, d)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # (N, r), (r,), (r, d)
    components = Vt[:k, :].T  # (d, k)
    explained_variance = s[:k] ** 2 / (X.shape[0] - 1)  # (k,)
    return components, explained_variance


# ---------------------------------------------------------------------------
# Attention as matrix operations
# ---------------------------------------------------------------------------


def softmax_rows(S: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the max-subtraction trick.

    Args:
        S: (T, T) scores (or any (..., n) array; softmax is over the last axis).
    Returns:
        (T, T) rows sum to one.
    """
    S_shift = S - S.max(axis=-1, keepdims=True)  # (T, T) subtract row max
    expS = np.exp(S_shift)  # (T, T)
    return expS / expS.sum(axis=-1, keepdims=True)  # (T, T)


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
    d_k = Q.shape[1]
    S = Q @ K.T / np.sqrt(d_k)  # (T, d_k) @ (d_k, T) -> (T, T)
    A = softmax_rows(S)  # (T, T), each row sums to 1
    Y = A @ V  # (T, T) @ (T, d_v) -> (T, d_v)
    return Y, A


def kronecker(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product ``A (x) B`` written out explicitly.

    ``(A (x) B)[i*p + k, j*q + l] = A[i, j] * B[k, l]``.

    Args:
        A: (m, n). B: (p, q).
    Returns:
        (m*p, n*q).
    """
    m, n = A.shape
    p, q = B.shape
    out = A[:, None, :, None] * B[None, :, None, :]  # (m, p, n, q)
    return out.reshape(m * p, n * q)  # (m*p, n*q)
