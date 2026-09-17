# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/pca.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k pca -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/pca --force

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
    raise NotImplementedError('TODO: implement center (see the reference in src/mlbook)')

def pca_eig(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by eigendecomposition of the covariance matrix.

    ``X``: (N, d) -> ``(components (d, r), explained_variance (r,), mean (d,))``.
    Components are sorted by decreasing variance; sign is fixed so the largest
    |entry| of each component is positive (eigenvectors are defined up to sign).
    Cost ``O(N d² + d³)``: fine when ``d`` is small, wasteful when ``d >> N``.
    """
    raise NotImplementedError('TODO: implement pca_eig (see the reference in src/mlbook)')

def pca_svd(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by thin SVD of the centred data (never forms the ``d × d`` covariance).

    ``X``: (N, d) -> ``(components (d, r), explained_variance (r,), mean (d,))``.
    Cost ``O(N d min(N, d))``; numerically better conditioned than ``pca_eig``
    because it never squares the singular values.
    """
    raise NotImplementedError('TODO: implement pca_svd (see the reference in src/mlbook)')

def transform(X: np.ndarray, V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Project: ``Z = (X - μ) V``. ``X``: (N, d), ``V``: (d, r) -> (N, r)."""
    raise NotImplementedError('TODO: implement transform (see the reference in src/mlbook)')

def inverse_transform(Z: np.ndarray, V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Reconstruct: ``X̂ = Z V^T + μ``. ``Z``: (N, r), ``V``: (d, r) -> (N, d)."""
    raise NotImplementedError('TODO: implement inverse_transform (see the reference in src/mlbook)')

def explained_variance_ratio(var: np.ndarray, total_var: float) -> np.ndarray:
    """``var_i / total``. ``var``: (r,) -> (r,). ``total_var`` = trace of the covariance."""
    raise NotImplementedError('TODO: implement explained_variance_ratio (see the reference in src/mlbook)')

def whiten(X: np.ndarray, V: np.ndarray, var: np.ndarray, mu: np.ndarray, eps: float=1e-08) -> np.ndarray:
    """PCA whitening: ``Z = (X - μ) V diag(1/√var)`` so ``cov(Z) = I``.

    ``X``: (N, d), ``V``: (d, r), ``var``: (r,) -> (N, r).
    """
    raise NotImplementedError('TODO: implement whiten (see the reference in src/mlbook)')

def randomized_svd(A: np.ndarray, rank: int, n_oversample: int=10, n_power_iters: int=2, seed: int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko–Martinsson–Tropp randomized range finder + small SVD.

    1. Sketch: ``Y = A Ω`` with Gaussian ``Ω`` (d, r + p);
    2. power iterations ``Y ← A (A^T Y)`` sharpen the spectrum;
    3. orthonormalise ``Q = qr(Y)``; 4. ``B = Q^T A`` is small; SVD it;
    5. ``U = Q Ũ``.

    ``A``: (N, d) -> ``(U (N, r), s (r,), Vt (r, d))``. Cost ``O(N d (r + p))``
    versus ``O(N d min(N, d))`` for a full SVD.
    """
    raise NotImplementedError('TODO: implement randomized_svd (see the reference in src/mlbook)')
