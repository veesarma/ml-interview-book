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


def intrinsics(fx: float, fy: float, cx: float, cy: float, skew: float = 0.0) -> np.ndarray:
    """Build ``K`` (3, 3)."""
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])  # (3, 3)


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: ``R = I + sinθ [k]_× + (1 − cosθ) [k]_ײ`` for unit axis ``k``. Returns (3, 3)."""
    k = axis / np.linalg.norm(axis)  # (3,)
    K = skew_symmetric(k)  # (3, 3)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)  # (3, 3)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """``[v]_×`` such that ``[v]_× w = v × w``. (3,) → (3, 3)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])  # (3, 3)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> tuple[np.ndarray, np.ndarray]:
    """Extrinsics for a camera at ``eye`` looking at ``target`` (z forward, y down).

    Returns:
        R (3, 3), t (3,) with ``P_c = R P_w + t``.
    """
    forward = target - eye  # (3,)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)  # (3,)  x axis = forward × up (y points down)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)  # (3,)
    R = np.stack([right, down, forward], axis=0)  # (3, 3) rows = camera axes in world coords
    t = -R @ eye  # (3,)
    return R, t


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``P = K [R | t]`` (3, 4)."""
    return K @ np.concatenate([R, t[:, None]], axis=1)  # (3, 4)


def to_homogeneous(x: np.ndarray) -> np.ndarray:
    """(N, d) → (N, d+1) by appending ones."""
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)  # (N, d+1)


def from_homogeneous(x: np.ndarray) -> np.ndarray:
    """(N, d+1) → (N, d) by dividing by the last coordinate."""
    return x[:, :-1] / x[:, -1:]  # (N, d)


def distort(xn: np.ndarray, dist: tuple[float, float, float, float, float]) -> np.ndarray:
    """Brown–Conrady distortion on *normalised* coordinates.

    ``r² = x² + y²``;  radial ``(1 + k₁r² + k₂r⁴ + k₃r⁶)``;  tangential
    ``x += 2p₁xy + p₂(r² + 2x²)``,  ``y += p₁(r² + 2y²) + 2p₂xy``.

    Args:
        xn: (N, 2) normalised (undistorted).  dist: (k1, k2, p1, p2, k3).
    Returns:
        (N, 2) distorted normalised coordinates.
    """
    k1, k2, p1, p2, k3 = dist
    x, y = xn[:, 0], xn[:, 1]  # (N,) each
    r2 = x * x + y * y  # (N,)
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3  # (N,)
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)  # (N,)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y  # (N,)
    return np.stack([xd, yd], axis=1)  # (N, 2)


def undistort(xd: np.ndarray, dist: tuple[float, float, float, float, float], iters: int = 10) -> np.ndarray:
    """Invert :func:`distort` by fixed-point iteration ``x ← x_d − (distort(x) − x)``. (N, 2) → (N, 2)."""
    x = xd.copy()  # (N, 2) initial guess: no distortion
    for _ in range(iters):
        x = xd - (distort(x, dist) - x)  # (N, 2)
    return x


def project(P_w: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Full pinhole projection ``p ~ K [R | t] P``.

    Steps: world→camera ``P_c = R P_w + t``; perspective divide ``x_n = (X/Z, Y/Z)``;
    optional distortion; pixels ``u = f_x x + c_x``, ``v = f_y y + c_y``.

    Args:
        P_w: (N, 3).
    Returns:
        pixels (N, 2), depths (N,) = Z in the camera frame (points with Z ≤ 0 are behind the camera).
    """
    P_c = P_w @ R.T + t[None, :]  # (N, 3) row-major: (R P)ᵀ = Pᵀ Rᵀ
    depth = P_c[:, 2]  # (N,)
    x_n = P_c[:, :2] / depth[:, None]  # (N, 2) normalised image coordinates
    if dist is not None:
        x_n = distort(x_n, dist)  # (N, 2)
    pixels = x_n @ K[:2, :2].T + K[:2, 2][None, :]  # (N, 2)  u = f_x x + s y + c_x, v = f_y y + c_y
    return pixels, depth


def unproject(pixels: np.ndarray, depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, dist: tuple | None = None) -> np.ndarray:
    """Inverse of :func:`project` given depth: pixel + Z → world point.

    ``x_n = K⁻¹ [u, v, 1]ᵀ`` (undistorted if needed), ``P_c = Z · [x_n, 1]``, ``P_w = Rᵀ (P_c − t)``.

    Args:
        pixels: (N, 2).  depth: (N,).
    Returns:
        (N, 3) world points.
    """
    K_inv = np.linalg.inv(K)  # (3, 3)
    rays = to_homogeneous(pixels) @ K_inv.T  # (N, 3) = (K⁻¹ p)ᵀ, third coordinate is 1
    x_n = rays[:, :2]  # (N, 2)
    if dist is not None:
        x_n = undistort(x_n, dist)  # (N, 2)
    P_c = np.concatenate([x_n, np.ones((len(x_n), 1))], axis=1) * depth[:, None]  # (N, 3)
    return (P_c - t[None, :]) @ R  # (N, 3) = (Rᵀ (P_c − t))ᵀ
