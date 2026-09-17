"""Triangulation (DLT) and PnP (DLT) — linear least-squares solutions to 3-D from 2-D (NumPy)."""

from __future__ import annotations

import numpy as np

from .camera import from_homogeneous, to_homogeneous


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Linear triangulation: for each correspondence solve ``A X = 0`` with ``A`` (4, 4).

    From ``p ~ P X`` the cross product ``p × (P X) = 0`` gives two independent rows per view:
        ``u · P[2] − P[0]``,  ``v · P[2] − P[1]``.
    The solution is the right singular vector of ``A`` with the smallest singular value.

    Args:
        P1, P2: (3, 4) projection matrices.  pts1, pts2: (N, 2) pixels.
    Returns:
        (N, 3) world points.
    """
    N = len(pts1)
    X = np.empty((N, 3))  # (N, 3)
    for i in range(N):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        A = np.stack([u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]], axis=0)  # (4, 4)
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]  # (4,) homogeneous solution
        X[i] = X_h[:3] / X_h[3]
    return X


def triangulate_multiview(Ps: list[np.ndarray], pixels: np.ndarray) -> np.ndarray:
    """DLT with ``V ≥ 2`` views for one point.  ``Ps``: list of (3, 4); ``pixels``: (V, 2). Returns (3,)."""
    rows = []
    for P, (u, v) in zip(Ps, pixels):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows, axis=0)  # (2V, 4)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]  # (4,)
    return X_h[:3] / X_h[3]


def reprojection_error(P: np.ndarray, X: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Per-point ``‖π(P X) − p‖₂``.  ``X``: (N, 3), ``pixels``: (N, 2) → (N,)."""
    proj = from_homogeneous(to_homogeneous(X) @ P.T)  # (N, 2)
    return np.linalg.norm(proj - pixels, axis=1)  # (N,)


def pnp_dlt(X: np.ndarray, pixels: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-n-Point by DLT: recover ``[R | t]`` from ≥ 6 3-D↔2-D correspondences.

    Work in normalised coordinates ``x_n = K⁻¹ p`` so the unknown is the (3, 4) matrix
    ``M = [R | t]`` up to scale; each point gives two rows of ``A m = 0`` (12 unknowns).
    The linear solution ignores orthonormality, so ``R`` is projected onto SO(3) via SVD
    and ``t`` rescaled consistently.  EPnP / iterative refinement improve on this.

    Args:
        X: (N, 3) world points.  pixels: (N, 2).  K: (3, 3).
    Returns:
        R (3, 3), t (3,).
    """
    x_n = to_homogeneous(pixels) @ np.linalg.inv(K).T  # (N, 3)
    X_h = to_homogeneous(X)  # (N, 4)
    zeros = np.zeros_like(X_h)  # (N, 4)
    rows_u = np.concatenate([X_h, zeros, -x_n[:, 0:1] * X_h], axis=1)  # (N, 12)
    rows_v = np.concatenate([zeros, X_h, -x_n[:, 1:2] * X_h], axis=1)  # (N, 12)
    A = np.concatenate([rows_u, rows_v], axis=0)  # (2N, 12)
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 4)  # (3, 4) up to scale and sign
    R_raw, t_raw = M[:, :3], M[:, 3]  # (3, 3), (3,)
    U, S, Vt_r = np.linalg.svd(R_raw)
    scale = S.mean()  # the singular values of a rotation are all 1
    R = U @ Vt_r  # (3, 3) nearest orthogonal matrix
    if np.linalg.det(R) < 0:  # fix reflection: rotations have det +1
        R, scale = -R, -scale
    t = t_raw / scale  # (3,)
    # points must be in front of the camera; flip the overall sign otherwise
    if np.mean((X @ R.T + t)[:, 2]) < 0:
        R, t = -R, -t
        R = R @ np.diag([1.0, 1.0, 1.0]) if np.linalg.det(R) > 0 else -R
    return R, t
