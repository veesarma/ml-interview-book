# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/camera.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k camera -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/camera --force

"""Pinhole camera model (NumPy): intrinsics, extrinsics, projection, unprojection, distortion.

Conventions (OpenCV-style):
* World points ``P_w ∈ ℝ³``; camera frame has +z forward, +x right, +y down.
* Extrinsics ``[R | t]`` map world → camera: ``P_c = R P_w + t``.
* Intrinsics ``K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]`` map normalised
  coordinates ``(x/z, y/z)`` → pixels ``(u, v)``.
* Points are stored row-wise: ``(N, 3)`` world points, ``(N, 2)`` pixels.
"""
from __future__ import annotations
import numpy as np

def intrinsics(fx: float, fy: float, cx: float, cy: float, skew: float=0.0) -> np.ndarray:
    """Build ``K`` (3, 3)."""
    raise NotImplementedError('TODO: implement intrinsics (see the reference in src/mlbook)')

def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: ``R = I + sinθ [k]_× + (1 − cosθ) [k]_ײ`` for unit axis ``k``. Returns (3, 3)."""
    raise NotImplementedError('TODO: implement rotation_from_axis_angle (see the reference in src/mlbook)')

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """``[v]_×`` such that ``[v]_× w = v × w``. (3,) → (3, 3)."""
    raise NotImplementedError('TODO: implement skew_symmetric (see the reference in src/mlbook)')

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray=np.array([0.0, 0.0, 1.0])) -> tuple[np.ndarray, np.ndarray]:
    """Extrinsics for a camera at ``eye`` looking at ``target`` (z forward, y down).

    Returns:
        R (3, 3), t (3,) with ``P_c = R P_w + t``.
    """
    raise NotImplementedError('TODO: implement look_at (see the reference in src/mlbook)')

def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``P = K [R | t]`` (3, 4)."""
    raise NotImplementedError('TODO: implement projection_matrix (see the reference in src/mlbook)')

def to_homogeneous(x: np.ndarray) -> np.ndarray:
    """(N, d) → (N, d+1) by appending ones."""
    raise NotImplementedError('TODO: implement to_homogeneous (see the reference in src/mlbook)')

def from_homogeneous(x: np.ndarray) -> np.ndarray:
    """(N, d+1) → (N, d) by dividing by the last coordinate."""
    raise NotImplementedError('TODO: implement from_homogeneous (see the reference in src/mlbook)')

def distort(xn: np.ndarray, dist: tuple[float, float, float, float, float]) -> np.ndarray:
    """Brown–Conrady distortion on *normalised* coordinates.

    ``r² = x² + y²``;  radial ``(1 + k₁r² + k₂r⁴ + k₃r⁶)``;  tangential
    ``x += 2p₁xy + p₂(r² + 2x²)``,  ``y += p₁(r² + 2y²) + 2p₂xy``.

    Args:
        xn: (N, 2) normalised (undistorted).  dist: (k1, k2, p1, p2, k3).
    Returns:
        (N, 2) distorted normalised coordinates.
    """
    raise NotImplementedError('TODO: implement distort (see the reference in src/mlbook)')

def undistort(xd: np.ndarray, dist: tuple[float, float, float, float, float], iters: int=10) -> np.ndarray:
    """Invert :func:`distort` by fixed-point iteration ``x ← x_d − (distort(x) − x)``. (N, 2) → (N, 2)."""
    raise NotImplementedError('TODO: implement undistort (see the reference in src/mlbook)')

def project(P_w: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: tuple | None=None) -> tuple[np.ndarray, np.ndarray]:
    """Full pinhole projection ``p ~ K [R | t] P``.

    Steps: world→camera ``P_c = R P_w + t``; perspective divide ``x_n = (X/Z, Y/Z)``;
    optional distortion; pixels ``u = f_x x + c_x``, ``v = f_y y + c_y``.

    Args:
        P_w: (N, 3).
    Returns:
        pixels (N, 2), depths (N,) = Z in the camera frame (points with Z ≤ 0 are behind the camera).
    """
    raise NotImplementedError('TODO: implement project (see the reference in src/mlbook)')

def unproject(pixels: np.ndarray, depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: tuple | None=None) -> np.ndarray:
    """Inverse of :func:`project` given depth: pixel + Z → world point.

    ``x_n = K⁻¹ [u, v, 1]ᵀ`` (undistorted if needed), ``P_c = Z · [x_n, 1]``, ``P_w = Rᵀ (P_c − t)``.

    Args:
        pixels: (N, 2).  depth: (N,).
    Returns:
        (N, 3) world points.
    """
    raise NotImplementedError('TODO: implement unproject (see the reference in src/mlbook)')
