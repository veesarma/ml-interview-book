# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/icp.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k icp -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/icp --force

"""Aligning two point clouds: closed-form Umeyama, then ICP when correspondences are unknown.

Two jobs in an offboard aerial stack need this.  First, a reconstruction from images alone
is fixed only up to a similarity (rotation, translation, scale), so comparing it against
survey-grade lidar or against yesterday's model means solving for that similarity.  Second,
a feed-forward network that predicts a pointmap predicts it in its own frame and usually at
its own scale, so the same alignment is what turns the prediction into metres.

Two regimes:

* **Correspondences known** (matched feature tracks, checkerboard corners, GPS-tagged
  control points): Umeyama gives the optimal similarity in closed form, one SVD, no
  iteration, no initialisation.
* **Correspondences unknown** (two raw clouds): ICP alternates "assign nearest neighbour"
  with "solve the closed form", which is coordinate descent on a non-convex cost.  It
  converges to a local minimum, so the initial guess is part of the algorithm.

All clouds are ``(N, 3)`` rows.  A similarity maps source to target as ``s R x + t``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def kabsch_umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool=False) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimal similarity mapping ``src`` onto ``dst`` for *known* correspondences.

    Minimises $\\sum_i \\| s R x_i + t - y_i \\|^2$.  Centring removes $t$, the rotation is the
    orthogonal Procrustes solution $R = U \\,\\mathrm{diag}(1,1,\\det(UV^\\top))\\, V^\\top$ of the
    cross-covariance, and the reflection guard is what keeps $R$ in SO(3) instead of O(3):
    a mirrored "fit" can have lower error and is physically meaningless.

    Args:
        src: (N, 3) source points.  dst: (N, 3) targets, row i matching row i of ``src``.
        with_scale: solve for ``s`` as well (Umeyama).  Leave ``False`` for a rigid fit.
    Returns:
        R (3, 3), t (3,), s (float) with ``dst ≈ s R src + t``.
    """
    raise NotImplementedError('TODO: implement kabsch_umeyama (see the reference in src/mlbook)')

def apply_similarity(X: np.ndarray, R: np.ndarray, t: np.ndarray, s: float=1.0) -> np.ndarray:
    """``s R X + t`` for row-major points. (N, 3) -> (N, 3)."""
    raise NotImplementedError('TODO: implement apply_similarity (see the reference in src/mlbook)')

def nearest_neighbours(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force nearest neighbour of every ``src`` point in ``dst``.

    ``O(N M)`` and deliberately so: a KD-tree is the right production answer, and writing
    the brute-force version keeps the ICP loop readable.  Swap in ``scipy.spatial.cKDTree``
    when ``N M`` stops fitting in memory.

    Returns:
        idx (N,) index into ``dst``, dist (N,) Euclidean distance.
    """
    raise NotImplementedError('TODO: implement nearest_neighbours (see the reference in src/mlbook)')

def estimate_normals(points: np.ndarray, k: int=8) -> np.ndarray:
    """Per-point surface normal from the local PCA of its ``k`` nearest neighbours. (N,3)->(N,3).

    The smallest-eigenvalue eigenvector of the neighbourhood covariance is the direction of
    least spread, which on a locally planar surface is the normal.  Signs are left
    unoriented, which is fine for point-to-plane ICP because the residual is squared.
    """
    raise NotImplementedError('TODO: implement estimate_normals (see the reference in src/mlbook)')

@dataclass
class ICPResult:
    """Transform plus the diagnostics you need to decide whether to believe it."""
    R: np.ndarray
    t: np.ndarray
    s: float
    rmse: float
    n_iters: int
    inlier_frac: float
    converged: bool

def icp(src: np.ndarray, dst: np.ndarray, max_iters: int=50, tol: float=1e-07, trim: float=1.0, with_scale: bool=False, init: tuple[np.ndarray, np.ndarray, float] | None=None) -> ICPResult:
    """Point-to-point ICP: alternate nearest-neighbour assignment and the closed-form fit.

    Args:
        src, dst: (N, 3) and (M, 3); no correspondence assumed.
        trim: keep this fraction of the best-matching source points each iteration
            (trimmed ICP).  Below 1.0 it tolerates partial overlap, which is the usual
            case when one cloud covers a larger footprint than the other.
        init: optional ``(R, t, s)`` starting guess, e.g. from GPS/IMU metadata.

    Returns:
        ``ICPResult`` with the accumulated transform taking ``src`` into ``dst``'s frame.
    """
    raise NotImplementedError('TODO: implement icp (see the reference in src/mlbook)')

def icp_point_to_plane(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray | None=None, max_iters: int=30, tol: float=1e-08) -> ICPResult:
    """ICP minimising distance to the target's tangent plane rather than to its points.

    The residual is $n_i^\\top (R p_i + t - q_i)$.  Linearising $R \\approx I + [\\omega]_\\times$
    turns each correspondence into one linear equation in the 6-vector $(\\omega, t)$:

    $$(p_i \\times n_i)^\\top \\omega + n_i^\\top t = -n_i^\\top (p_i - q_i).$$

    On terrain this converges in a handful of iterations where point-to-point crawls,
    because a point sliding along a flat roof or field costs nothing in this metric.
    """
    raise NotImplementedError('TODO: implement icp_point_to_plane (see the reference in src/mlbook)')
