"""PCA via covariance eigendecomposition and via SVD, plus whitening and a
randomized SVD for large matrices.

With centred ``X`` (N, d): ``X = U S V^T``, covariance ``C = X^T X / (N-1)
= V (S²/(N-1)) V^T``. The principal directions are the columns of ``V`` (the
eigenvectors of ``C``), the variance along direction ``i`` is ``s_i² / (N-1)``,
and the projection is ``Z = X V_r = U_r S_r`` (N, r).
"""

from __future__ import annotations

import numpy as np


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X - mean, mean)``. ``X``: (N, d) -> ((N, d), (d,))."""
    mu = X.mean(axis=0)  # (d,)
    return X - mu, mu


def pca_eig(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by eigendecomposition of the covariance matrix.

    ``X``: (N, d) -> ``(components (d, r), explained_variance (r,), mean (d,))``.
    Components are sorted by decreasing variance; sign is fixed so the largest
    |entry| of each component is positive (eigenvectors are defined up to sign).
    Cost ``O(N d² + d³)``: fine when ``d`` is small, wasteful when ``d >> N``.
    """
    Xc, mu = center(X)
    C = Xc.T @ Xc / (X.shape[0] - 1)  # (d, d)
    evals, evecs = np.linalg.eigh(C)  # (d,), (d, d) ascending
    order = np.argsort(evals)[::-1][:n_components]  # (r,)
    V = evecs[:, order]  # (d, r)
    V = V * np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])[None, :]  # (d, r) sign fix
    return V, evals[order], mu


def pca_svd(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by thin SVD of the centred data (never forms the ``d × d`` covariance).

    ``X``: (N, d) -> ``(components (d, r), explained_variance (r,), mean (d,))``.
    Cost ``O(N d min(N, d))``; numerically better conditioned than ``pca_eig``
    because it never squares the singular values.
    """
    Xc, mu = center(X)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # s: (min(N,d),), Vt: (min(N,d), d)
    V = Vt[:n_components].T  # (d, r)
    V = V * np.sign(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])])[None, :]  # (d, r) sign fix
    var = s[:n_components] ** 2 / (X.shape[0] - 1)  # (r,)
    return V, var, mu


def transform(X: np.ndarray, V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Project: ``Z = (X - μ) V``. ``X``: (N, d), ``V``: (d, r) -> (N, r)."""
    return (X - mu) @ V  # (N, r)


def inverse_transform(Z: np.ndarray, V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Reconstruct: ``X̂ = Z V^T + μ``. ``Z``: (N, r), ``V``: (d, r) -> (N, d)."""
    return Z @ V.T + mu  # (N, d)


def explained_variance_ratio(var: np.ndarray, total_var: float) -> np.ndarray:
    """``var_i / total``. ``var``: (r,) -> (r,). ``total_var`` = trace of the covariance."""
    return var / total_var


def whiten(X: np.ndarray, V: np.ndarray, var: np.ndarray, mu: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """PCA whitening: ``Z = (X - μ) V diag(1/√var)`` so ``cov(Z) = I``.

    ``X``: (N, d), ``V``: (d, r), ``var``: (r,) -> (N, r).
    """
    return transform(X, V, mu) / np.sqrt(var + eps)[None, :]  # (N, r)


def randomized_svd(
    A: np.ndarray, rank: int, n_oversample: int = 10, n_power_iters: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko–Martinsson–Tropp randomized range finder + small SVD.

    1. Sketch: ``Y = A Ω`` with Gaussian ``Ω`` (d, r + p);
    2. power iterations ``Y ← A (A^T Y)`` sharpen the spectrum;
    3. orthonormalise ``Q = qr(Y)``; 4. ``B = Q^T A`` is small; SVD it;
    5. ``U = Q Ũ``.

    ``A``: (N, d) -> ``(U (N, r), s (r,), Vt (r, d))``. Cost ``O(N d (r + p))``
    versus ``O(N d min(N, d))`` for a full SVD.
    """
    rng = np.random.default_rng(seed)
    N, d = A.shape
    Omega = rng.standard_normal((d, rank + n_oversample))  # (d, r+p)
    Y = A @ Omega  # (N, r+p)
    for _ in range(n_power_iters):
        Y = A @ (A.T @ Y)  # (N, r+p)
        Y, _ = np.linalg.qr(Y)  # re-orthonormalise for stability
    Q, _ = np.linalg.qr(Y)  # (N, r+p)
    B = Q.T @ A  # (r+p, d)
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)  # (r+p, r+p), (r+p,), (r+p, d)
    U = Q @ Ub  # (N, r+p)
    return U[:, :rank], s[:rank], Vt[:rank]
