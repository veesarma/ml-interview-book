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

    model: Any                 # the fitted model (F, H, or (R, t))
    inliers: np.ndarray        # (N,) bool mask
    residuals: np.ndarray      # (N,) residual of every datum under ``model``
    n_iters: int               # iterations actually run (adaptive stopping)

    @property
    def inlier_ratio(self) -> float:
        return float(self.inliers.mean())


def required_iterations(
    inlier_ratio: float, sample_size: int, confidence: float = 0.99, max_iters: int = 100_000
) -> int:
    r"""Iterations needed to draw one all-inlier sample with probability ``confidence``.

    One sample is all-inlier with probability $w^s$, so $k$ samples all fail with
    probability $(1 - w^s)^k$.  Demanding that this stay below $1 - p$ gives

    $$k \ge \frac{\log(1 - p)}{\log(1 - w^s)}.$$

    The count is exponential in ``sample_size``: at $w = 0.5$ a homography (s=4) needs
    72 iterations and a fundamental matrix (s=8) needs 1177.  That is the whole argument
    for minimal solvers.
    """
    w = float(np.clip(inlier_ratio, 1e-6, 1.0 - 1e-12))
    if w >= 1.0 - 1e-12:
        return 1
    denom = np.log1p(-(w**sample_size))  # log(1 - w^s), accurate for small w^s
    if denom >= -1e-15:                   # w^s underflowed: the model is hopeless
        return max_iters
    return int(min(max_iters, np.ceil(np.log(1.0 - confidence) / denom)))


def ransac(
    n_data: int,
    sample_size: int,
    fit_fn: Callable[[np.ndarray], Any],
    residual_fn: Callable[[Any], np.ndarray],
    threshold: float,
    confidence: float = 0.99,
    max_iters: int = 2000,
    refit: bool = True,
    rng: np.random.Generator | None = None,
) -> RansacResult:
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
    rng = np.random.default_rng() if rng is None else rng
    best_model, best_inliers, best_score = None, np.zeros(n_data, bool), -1.0
    budget, it = max_iters, 0
    while it < budget:
        it += 1
        idx = rng.choice(n_data, size=sample_size, replace=False)  # (s,)
        model = fit_fn(idx)
        if model is None:
            continue
        inliers = residual_fn(model) < threshold  # (N,) bool
        score = float(inliers.sum())
        if score > best_score:
            best_model, best_inliers, best_score = model, inliers, score
            # adaptive stopping: the better the current consensus, the fewer draws left
            budget = min(max_iters, required_iterations(score / n_data, sample_size, confidence))
    if best_model is None:
        return RansacResult(None, best_inliers, np.full(n_data, np.inf), it)
    if refit and best_inliers.sum() >= sample_size:
        refined = fit_fn(np.flatnonzero(best_inliers))
        if refined is not None:
            residuals = residual_fn(refined)
            if (residuals < threshold).sum() >= best_inliers.sum():
                best_model, best_inliers = refined, residuals < threshold
    return RansacResult(best_model, best_inliers, residual_fn(best_model), it)


# --------------------------------------------------------------------------- models


def fit_homography_dlt(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray | None:
    """Normalised DLT homography ``x2 ~ H x1`` from >= 4 correspondences. (N,2),(N,2) -> (3,3).

    Each correspondence gives two rows of ``A h = 0`` from ``x2 x (H x1) = 0``; ``h`` is
    the right singular vector of the smallest singular value.  Hartley normalisation
    (centre, scale to mean distance sqrt(2)) is what makes the linear solve well conditioned.
    """
    if len(pts1) < 4:
        return None
    from .epipolar import normalise_points

    x1, T1 = normalise_points(pts1)  # (N, 3) normalised, (3, 3)
    x2, T2 = normalise_points(pts2)  # (N, 3), (3, 3)
    if not (np.isfinite(x1).all() and np.isfinite(x2).all()):
        return None  # degenerate sample: every point identical, so the scale blew up
    zeros = np.zeros((len(pts1), 3))  # (N, 3)
    row_u = np.concatenate([zeros, -x2[:, 2:3] * x1, x2[:, 1:2] * x1], axis=1)  # (N, 9)
    row_v = np.concatenate([x2[:, 2:3] * x1, zeros, -x2[:, 0:1] * x1], axis=1)  # (N, 9)
    A = np.concatenate([row_u, row_v], axis=0)  # (2N, 9)
    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)  # (3, 3) in normalised coordinates
    H = np.linalg.inv(T2) @ H_n @ T1  # (3, 3) undo both normalisations
    if abs(H[2, 2]) < 1e-12 or not np.isfinite(H).all():
        return None
    return H / H[2, 2]


def transfer_error(H: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Symmetric transfer error ``||H x1 - x2|| + ||H^-1 x2 - x1||`` in pixels. -> (N,)."""
    fwd = from_homogeneous(to_homogeneous(pts1) @ H.T)      # (N, 2)
    bwd = from_homogeneous(to_homogeneous(pts2) @ np.linalg.inv(H).T)  # (N, 2)
    return np.linalg.norm(fwd - pts2, axis=1) + np.linalg.norm(bwd - pts1, axis=1)  # (N,)


def ransac_homography(
    pts1: np.ndarray, pts2: np.ndarray, threshold: float = 3.0, **kw
) -> RansacResult:
    """Robust ``H`` from putative matches. Minimal sample 4."""
    return ransac(
        n_data=len(pts1),
        sample_size=4,
        fit_fn=lambda idx: fit_homography_dlt(pts1[idx], pts2[idx]),
        residual_fn=lambda H: transfer_error(H, pts1, pts2),
        threshold=threshold,
        **kw,
    )


def ransac_fundamental(
    pts1: np.ndarray, pts2: np.ndarray, threshold: float = 1.0, **kw
) -> RansacResult:
    """Robust ``F`` from putative matches, scored by Sampson distance. Minimal sample 8.

    The 8-point algorithm is used rather than the 5-point (calibrated) or 7-point solver
    because it is linear; the cost is a larger minimal sample, hence more iterations.
    """
    return ransac(
        n_data=len(pts1),
        sample_size=8,
        fit_fn=lambda idx: eight_point(pts1[idx], pts2[idx]),
        residual_fn=lambda F: sampson_distance(F, pts1, pts2),
        threshold=threshold,
        **kw,
    )


def ransac_pnp(
    X: np.ndarray, pixels: np.ndarray, K: np.ndarray, threshold: float = 3.0, **kw
) -> RansacResult:
    """Robust camera pose from 3-D to 2-D correspondences, scored by reprojection error.

    Args:
        X: (N, 3) world points.  pixels: (N, 2).  K: (3, 3).
    Returns:
        ``RansacResult`` whose ``model`` is ``(R (3,3), t (3,))``.
    """

    def fit(idx: np.ndarray):
        if len(idx) < 6:
            return None
        try:
            return pnp_dlt(X[idx], pixels[idx], K)
        except np.linalg.LinAlgError:
            return None

    def residual(model) -> np.ndarray:
        R, t = model
        return reprojection_error(projection_matrix(K, R, t), X, pixels)  # (N,)

    return ransac(len(X), 6, fit, residual, threshold, **kw)
