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


def kabsch_umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Optimal similarity mapping ``src`` onto ``dst`` for *known* correspondences.

    Minimises $\sum_i \| s R x_i + t - y_i \|^2$.  Centring removes $t$, the rotation is the
    orthogonal Procrustes solution $R = U \,\mathrm{diag}(1,1,\det(UV^\top))\, V^\top$ of the
    cross-covariance, and the reflection guard is what keeps $R$ in SO(3) instead of O(3):
    a mirrored "fit" can have lower error and is physically meaningless.

    Args:
        src: (N, 3) source points.  dst: (N, 3) targets, row i matching row i of ``src``.
        with_scale: solve for ``s`` as well (Umeyama).  Leave ``False`` for a rigid fit.
    Returns:
        R (3, 3), t (3,), s (float) with ``dst ≈ s R src + t``.
    """
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)     # (3,), (3,)
    Xs, Xd = src - mu_s, dst - mu_d                     # (N, 3), (N, 3) centred
    H = (Xd.T @ Xs) / len(src)                          # (3, 3) cross-covariance
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:        # guard against a reflection
        S[2, 2] = -1.0
    R = U @ S @ Vt                                      # (3, 3) in SO(3)
    s = float(np.trace(np.diag(D) @ S) / np.mean(np.sum(Xs**2, axis=1))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s                             # (3,)
    return R, t, s


def apply_similarity(X: np.ndarray, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    """``s R X + t`` for row-major points. (N, 3) -> (N, 3)."""
    return s * (X @ R.T) + t  # (N, 3)


def nearest_neighbours(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force nearest neighbour of every ``src`` point in ``dst``.

    ``O(N M)`` and deliberately so: a KD-tree is the right production answer, and writing
    the brute-force version keeps the ICP loop readable.  Swap in ``scipy.spatial.cKDTree``
    when ``N M`` stops fitting in memory.

    Returns:
        idx (N,) index into ``dst``, dist (N,) Euclidean distance.
    """
    d2 = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)  # (N, M) squared distances
    idx = np.argmin(d2, axis=1)                                   # (N,)
    return idx, np.sqrt(d2[np.arange(len(src)), idx])             # (N,), (N,)


def estimate_normals(points: np.ndarray, k: int = 8) -> np.ndarray:
    """Per-point surface normal from the local PCA of its ``k`` nearest neighbours. (N,3)->(N,3).

    The smallest-eigenvalue eigenvector of the neighbourhood covariance is the direction of
    least spread, which on a locally planar surface is the normal.  Signs are left
    unoriented, which is fine for point-to-plane ICP because the residual is squared.
    """
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)  # (N, N)
    order = np.argsort(d2, axis=1)[:, : k + 1]                          # (N, k+1) self first
    normals = np.empty_like(points)                                     # (N, 3)
    for i, nb in enumerate(order):
        patch = points[nb] - points[nb].mean(axis=0)                    # (k+1, 3)
        _, _, Vt = np.linalg.svd(patch, full_matrices=False)
        normals[i] = Vt[-1]                                             # (3,) least-variance axis
    return normals


@dataclass
class ICPResult:
    """Transform plus the diagnostics you need to decide whether to believe it."""

    R: np.ndarray            # (3, 3)
    t: np.ndarray            # (3,)
    s: float                 # scale (1.0 for rigid)
    rmse: float              # RMS residual over the correspondences used at the last step
    n_iters: int
    inlier_frac: float       # fraction of source points kept after trimming
    converged: bool


def icp(
    src: np.ndarray,
    dst: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-7,
    trim: float = 1.0,
    with_scale: bool = False,
    init: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> ICPResult:
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
    R_acc = np.eye(3) if init is None else init[0].copy()   # (3, 3)
    t_acc = np.zeros(3) if init is None else init[1].copy()  # (3,)
    s_acc = 1.0 if init is None else init[2]
    prev_rmse, converged = np.inf, False
    keep = max(3, int(round(trim * len(src))))
    it = 0
    for it in range(1, max_iters + 1):
        moved = apply_similarity(src, R_acc, t_acc, s_acc)   # (N, 3) current estimate
        idx, dist = nearest_neighbours(moved, dst)           # (N,), (N,)
        sel = np.argsort(dist)[:keep]                        # (keep,) trimmed correspondences
        R_step, t_step, s_step = kabsch_umeyama(moved[sel], dst[idx[sel]], with_scale)
        # compose the incremental fit onto the accumulated one
        R_acc, t_acc = R_step @ R_acc, s_step * (R_step @ t_acc) + t_step
        s_acc *= s_step
        rmse = float(np.sqrt((dist[sel] ** 2).mean()))
        if abs(prev_rmse - rmse) < tol:
            converged = True
            break
        prev_rmse = rmse
    moved = apply_similarity(src, R_acc, t_acc, s_acc)
    _, dist = nearest_neighbours(moved, dst)
    final = float(np.sqrt((np.sort(dist)[:keep] ** 2).mean()))
    return ICPResult(R_acc, t_acc, s_acc, final, it, keep / len(src), converged)


def icp_point_to_plane(
    src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray | None = None,
    max_iters: int = 30, tol: float = 1e-8,
) -> ICPResult:
    r"""ICP minimising distance to the target's tangent plane rather than to its points.

    The residual is $n_i^\top (R p_i + t - q_i)$.  Linearising $R \approx I + [\omega]_\times$
    turns each correspondence into one linear equation in the 6-vector $(\omega, t)$:

    $$(p_i \times n_i)^\top \omega + n_i^\top t = -n_i^\top (p_i - q_i).$$

    On terrain this converges in a handful of iterations where point-to-point crawls,
    because a point sliding along a flat roof or field costs nothing in this metric.
    """
    normals = estimate_normals(dst) if dst_normals is None else dst_normals  # (M, 3)
    R_acc, t_acc, prev, converged = np.eye(3), np.zeros(3), np.inf, False
    it = 0
    for it in range(1, max_iters + 1):
        moved = src @ R_acc.T + t_acc                     # (N, 3)
        idx, dist = nearest_neighbours(moved, dst)        # (N,), (N,)
        q, n = dst[idx], normals[idx]                     # (N, 3), (N, 3)
        A = np.concatenate([np.cross(moved, n), n], axis=1)          # (N, 6) [p x n | n]
        b = -np.sum(n * (moved - q), axis=1)                          # (N,)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)                   # (6,)
        from .bundle_adjustment import so3_exp
        dR = so3_exp(sol[:3])                             # (3, 3)
        R_acc, t_acc = dR @ R_acc, dR @ t_acc + sol[3:]
        rmse = float(np.sqrt((np.sum(n * (moved - q), axis=1) ** 2).mean()))
        if abs(prev - rmse) < tol:
            converged = True
            break
        prev = rmse
    moved = src @ R_acc.T + t_acc
    _, dist = nearest_neighbours(moved, dst)
    return ICPResult(R_acc, t_acc, 1.0, float(np.sqrt((dist**2).mean())), it, 1.0, converged)
