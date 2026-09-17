# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/triangulation.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k triangulation -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/triangulation --force

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
    raise NotImplementedError('TODO: implement triangulate_dlt (see the reference in src/mlbook)')

def triangulate_multiview(Ps: list[np.ndarray], pixels: np.ndarray) -> np.ndarray:
    """DLT with ``V ≥ 2`` views for one point.  ``Ps``: list of (3, 4); ``pixels``: (V, 2). Returns (3,)."""
    raise NotImplementedError('TODO: implement triangulate_multiview (see the reference in src/mlbook)')

def reprojection_error(P: np.ndarray, X: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Per-point ``‖π(P X) − p‖₂``.  ``X``: (N, 3), ``pixels``: (N, 2) → (N,)."""
    raise NotImplementedError('TODO: implement reprojection_error (see the reference in src/mlbook)')

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
    raise NotImplementedError('TODO: implement pnp_dlt (see the reference in src/mlbook)')
