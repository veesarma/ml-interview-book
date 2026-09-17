# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/epipolar.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k epipolar -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/epipolar --force

"""Two-view geometry (NumPy): fundamental / essential matrices, normalised 8-point, Sampson error.

Epipolar constraint for corresponding pixels ``p, p'`` (homogeneous, (3,)):
    p'ᵀ F p = 0,     F = K'⁻ᵀ E K⁻¹,     E = [t]_× R.
"""
from __future__ import annotations
import numpy as np
from .camera import skew_symmetric, to_homogeneous

def essential_from_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``E = [t]_× R`` for the relative pose taking camera-1 coordinates to camera-2. (3, 3)."""
    raise NotImplementedError('TODO: implement essential_from_pose (see the reference in src/mlbook)')

def fundamental_from_essential(E: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """``F = K₂⁻ᵀ E K₁⁻¹``. (3, 3)."""
    raise NotImplementedError('TODO: implement fundamental_from_essential (see the reference in src/mlbook)')

def normalise_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: translate to zero mean, scale so mean distance from origin is √2.

    Args:
        pts: (N, 2).
    Returns:
        normalised homogeneous points (N, 3) and the similarity ``T`` (3, 3) with ``x̂ = T x``.
    """
    raise NotImplementedError('TODO: implement normalise_points (see the reference in src/mlbook)')

def eight_point(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Normalised 8-point algorithm for ``F`` with ``pts2ᵀ F pts1 = 0``.

    Each correspondence gives one row ``[x'x, x'y, x', y'x, y'y, y', x, y, 1] · vec(F) = 0``;
    solve ``A f = 0`` by SVD (smallest singular vector), enforce rank 2 by zeroing the
    smallest singular value of ``F``, then undo the normalisation ``F = T₂ᵀ F̂ T₁``.

    Args:
        pts1, pts2: (N, 2) pixel correspondences, N ≥ 8.
    Returns:
        F (3, 3), scaled so ``‖F‖_F = 1``.
    """
    raise NotImplementedError('TODO: implement eight_point (see the reference in src/mlbook)')

def epipolar_lines(F: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """Lines ``l' = F p`` in image 2 for points in image 1. (N, 2) → (N, 3) as ``(a, b, c)``: ax + by + c = 0."""
    raise NotImplementedError('TODO: implement epipolar_lines (see the reference in src/mlbook)')

def sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """First-order geometric error ``(p'ᵀ F p)² / ((Fp)₁² + (Fp)₂² + (Fᵀp')₁² + (Fᵀp')₂²)``. (N,)."""
    raise NotImplementedError('TODO: implement sampson_distance (see the reference in src/mlbook)')

def epipoles(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right null vectors: ``F e = 0`` (epipole in image 1), ``Fᵀ e' = 0`` (image 2). Each (2,) in pixels."""
    raise NotImplementedError('TODO: implement epipoles (see the reference in src/mlbook)')
