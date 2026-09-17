"""Two-view geometry (NumPy): fundamental / essential matrices, normalised 8-point, Sampson error.

Epipolar constraint for corresponding pixels ``p, p'`` (homogeneous, (3,)):
    p'ᵀ F p = 0,     F = K'⁻ᵀ E K⁻¹,     E = [t]_× R.
"""

from __future__ import annotations

import numpy as np

from .camera import skew_symmetric, to_homogeneous


def essential_from_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``E = [t]_× R`` for the relative pose taking camera-1 coordinates to camera-2. (3, 3)."""
    return skew_symmetric(t) @ R  # (3, 3)


def fundamental_from_essential(E: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """``F = K₂⁻ᵀ E K₁⁻¹``. (3, 3)."""
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)  # (3, 3)


def normalise_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: translate to zero mean, scale so mean distance from origin is √2.

    Args:
        pts: (N, 2).
    Returns:
        normalised homogeneous points (N, 3) and the similarity ``T`` (3, 3) with ``x̂ = T x``.
    """
    mean = pts.mean(axis=0)  # (2,)
    mean_dist = np.mean(np.linalg.norm(pts - mean, axis=1))  # scalar
    scale = np.sqrt(2.0) / mean_dist
    T = np.array([[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]])  # (3, 3)
    return to_homogeneous(pts) @ T.T, T  # (N, 3), (3, 3)


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
    x1, T1 = normalise_points(pts1)  # (N, 3), (3, 3)
    x2, T2 = normalise_points(pts2)  # (N, 3), (3, 3)
    A = np.stack(
        [x2[:, 0] * x1[:, 0], x2[:, 0] * x1[:, 1], x2[:, 0], x2[:, 1] * x1[:, 0], x2[:, 1] * x1[:, 1], x2[:, 1], x1[:, 0], x1[:, 1], np.ones(len(x1))],
        axis=1,
    )  # (N, 9)
    _, _, Vt = np.linalg.svd(A)  # Vt: (9, 9)
    F_hat = Vt[-1].reshape(3, 3)  # (3, 3) null vector → matrix
    U, S, Vt_f = np.linalg.svd(F_hat)
    S[2] = 0.0  # enforce rank 2: all epipolar lines must meet at one epipole
    F_hat = U @ np.diag(S) @ Vt_f  # (3, 3)
    F = T2.T @ F_hat @ T1  # (3, 3) back to pixel coordinates
    return F / np.linalg.norm(F)  # (3, 3)


def epipolar_lines(F: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """Lines ``l' = F p`` in image 2 for points in image 1. (N, 2) → (N, 3) as ``(a, b, c)``: ax + by + c = 0."""
    return to_homogeneous(pts1) @ F.T  # (N, 3)


def sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """First-order geometric error ``(p'ᵀ F p)² / ((Fp)₁² + (Fp)₂² + (Fᵀp')₁² + (Fᵀp')₂²)``. (N,)."""
    x1 = to_homogeneous(pts1)  # (N, 3)
    x2 = to_homogeneous(pts2)  # (N, 3)
    Fx1 = x1 @ F.T  # (N, 3)  F p
    Ftx2 = x2 @ F  # (N, 3)  Fᵀ p'
    algebraic = np.sum(x2 * Fx1, axis=1)  # (N,)  p'ᵀ F p
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2  # (N,)
    return algebraic**2 / np.maximum(denom, 1e-12)  # (N,)


def epipoles(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left/right null vectors: ``F e = 0`` (epipole in image 1), ``Fᵀ e' = 0`` (image 2). Each (2,) in pixels."""
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]  # (3,)
    U, _, _ = np.linalg.svd(F)
    e2 = U[:, -1]  # (3,)
    return e1[:2] / e1[2], e2[:2] / e2[2]
