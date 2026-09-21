# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/ransac.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ransac -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/ransac --force

"""RANSAC: fitting geometry when a fraction of the correspondences are wrong (NumPy).

Aerial survey matching produces outliers by the hundreds: repeated roof tiles, moving
vegetation, water glint, a parked car that moved between passes.  A least-squares fit
over all matches is dragged arbitrarily far by a single gross error, because the
squared-error cost is unbounded.  RANSAC replaces "fit everything" with "fit a minimal
sample, count who agrees, keep the best consensus".

The loop, for a model needing ``s`` points:

1. sample ``s`` correspondences uniformly at random,
2. fit the model exactly (minimal solve),
3. score every correspondence with a geometric residual,
4. keep the model with the largest inlier set,
5. refit on all inliers at the end (the minimal fit was never the goal).

Conventions match ``mlbook.geometry.camera``: pixels are ``(N, 2)`` rows, world points
``(N, 3)`` rows, extrinsics map world to camera as ``P_c = R P_w + t``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from .camera import from_homogeneous, projection_matrix, to_homogeneous
from .epipolar import eight_point, sampson_distance
from .triangulation import pnp_dlt, reprojection_error

@dataclass
class RansacResult:
    """What a robust fit returns: the model, who voted for it, and how long it took."""
    model: Any
    inliers: np.ndarray
    residuals: np.ndarray
    n_iters: int

    @property
    def inlier_ratio(self) -> float:
        raise NotImplementedError('TODO: implement inlier_ratio (see the reference in src/mlbook)')

def required_iterations(inlier_ratio: float, sample_size: int, confidence: float=0.99, max_iters: int=100000) -> int:
    """Iterations needed to draw one all-inlier sample with probability ``confidence``.

    One sample is all-inlier with probability $w^s$, so $k$ samples all fail with
    probability $(1 - w^s)^k$.  Demanding that this stay below $1 - p$ gives

    $$k \\ge \\frac{\\log(1 - p)}{\\log(1 - w^s)}.$$

    The count is exponential in ``sample_size``: at $w = 0.5$ a homography (s=4) needs
    72 iterations and a fundamental matrix (s=8) needs 1177.  That is the whole argument
    for minimal solvers.
    """
    raise NotImplementedError('TODO: implement required_iterations (see the reference in src/mlbook)')

def ransac(n_data: int, sample_size: int, fit_fn: Callable[[np.ndarray], Any], residual_fn: Callable[[Any], np.ndarray], threshold: float, confidence: float=0.99, max_iters: int=2000, refit: bool=True, rng: np.random.Generator | None=None) -> RansacResult:
    """Generic RANSAC loop with adaptive stopping.

    Args:
        n_data: number of correspondences ``N``.
        sample_size: minimal sample ``s`` the solver needs.
        fit_fn: ``idx (s,) -> model``; may return ``None`` for a degenerate sample.
        residual_fn: ``model -> (N,)`` residuals in the units of ``threshold``.
        threshold: inlier cutoff, in the residual's units (pixels for all fits here).
        refit: refit on the final consensus set (always do this in practice).

    Returns:
        ``RansacResult``.  ``model`` is ``None`` only if every sample was degenerate.
    """
    raise NotImplementedError('TODO: implement ransac (see the reference in src/mlbook)')

def fit_homography_dlt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray | None:
    """Normalised DLT homography ``x2 ~ H x1`` from >= 4 correspondences. (N,2),(N,2) -> (3,3).

    Each correspondence gives two rows of ``A h = 0`` from ``x2 x (H x1) = 0``; ``h`` is
    the right singular vector of the smallest singular value.  Hartley normalisation
    (centre, scale to mean distance sqrt(2)) is what makes the linear solve well conditioned.
    """
    raise NotImplementedError('TODO: implement fit_homography_dlt (see the reference in src/mlbook)')

def transfer_error(H: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Symmetric transfer error ``||H x1 - x2|| + ||H^-1 x2 - x1||`` in pixels. -> (N,)."""
    raise NotImplementedError('TODO: implement transfer_error (see the reference in src/mlbook)')

def ransac_homography(pts1: np.ndarray, pts2: np.ndarray, threshold: float=3.0, **kw) -> RansacResult:
    """Robust ``H`` from putative matches. Minimal sample 4."""
    raise NotImplementedError('TODO: implement ransac_homography (see the reference in src/mlbook)')

def ransac_fundamental(pts1: np.ndarray, pts2: np.ndarray, threshold: float=1.0, **kw) -> RansacResult:
    """Robust ``F`` from putative matches, scored by Sampson distance. Minimal sample 8.

    The 8-point algorithm is used rather than the 5-point (calibrated) or 7-point solver
    because it is linear; the cost is a larger minimal sample, hence more iterations.
    """
    raise NotImplementedError('TODO: implement ransac_fundamental (see the reference in src/mlbook)')

def ransac_pnp(X: np.ndarray, pixels: np.ndarray, K: np.ndarray, threshold: float=3.0, **kw) -> RansacResult:
    """Robust camera pose from 3-D to 2-D correspondences, scored by reprojection error.

    Args:
        X: (N, 3) world points.  pixels: (N, 2).  K: (3, 3).
    Returns:
        ``RansacResult`` whose ``model`` is ``(R (3,3), t (3,))``.
    """
    raise NotImplementedError('TODO: implement ransac_pnp (see the reference in src/mlbook)')
