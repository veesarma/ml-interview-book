"""Incremental structure from motion: images in, camera poses and sparse 3-D out (NumPy).

The classical pipeline, and still the thing a learned reconstruction model is measured
against:

    features -> matches -> geometric verification (RANSAC) -> relative pose
             -> triangulation -> incremental registration (PnP) -> bundle adjustment

This module implements everything downstream of matching, on *tracks*: a track is one
scene point observed in several views, which is what a feature matcher plus transitive
match-chaining produces.  Feature extraction itself is a separate problem, so tracks are
the input here.

Two facts to keep in the room:

* Two views alone determine the reconstruction **up to scale**.  The baseline length is
  unobservable from images, so ``||t|| = 1`` is a convention.  Metric scale has to come
  from outside: GPS baselines, a laser altimeter, ground control points, or a known object.
* Decomposing the essential matrix gives **four** candidate poses.  Only one puts the
  scene in front of both cameras, and that cheirality test is the disambiguator.

Conventions follow ``mlbook.geometry.camera``: ``P_c = R P_w + t``, pixels are ``(N, 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bundle_adjustment import BAProblem, bundle_adjust
from .camera import projection_matrix
from .ransac import ransac_fundamental, ransac_pnp
from .triangulation import reprojection_error, triangulate_dlt, triangulate_multiview


def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """``E = K2^T F K1``, then projected back onto the essential manifold.

    A noisy ``F`` gives an ``E`` whose singular values are not ``(sigma, sigma, 0)``.  An
    essential matrix must have two equal non-zero singular values and one zero, so replace
    them.  Skipping this step is a common source of a pose that is subtly wrong everywhere.
    """
    E = K2.T @ F @ K1                                  # (3, 3)
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt           # (3, 3) nearest essential matrix


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four ``(R, t)`` candidates consistent with ``E = [t]_x R``.

    With ``E = U diag(1,1,0) V^T`` and ``W`` the 90-degree rotation about z, the candidates
    are ``(U W V^T, +u_3)``, ``(U W V^T, -u_3)``, ``(U W^T V^T, +u_3)``, ``(U W^T V^T, -u_3)``.
    They correspond to the true solution plus a twist of one camera by 180 degrees about the
    baseline and a sign flip of the baseline, both of which put points behind a camera.

    Returns:
        list of 4 ``(R (3,3), t (3,))`` with ``||t|| = 1``.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # (3, 3)
    R_a, R_b = U @ W @ Vt, U @ W.T @ Vt                                  # (3, 3) each
    t = U[:, 2]                                                          # (3,) unit baseline
    return [(R_a, t), (R_a, -t), (R_b, t), (R_b, -t)]


def _cheirality_count(R: np.ndarray, t: np.ndarray, K: np.ndarray,
                      pts1: np.ndarray, pts2: np.ndarray) -> tuple[int, np.ndarray]:
    """How many triangulated points sit in front of *both* cameras, and where they are."""
    P1 = projection_matrix(K, np.eye(3), np.zeros(3))   # (3, 4) first camera is the origin
    P2 = projection_matrix(K, R, t)                     # (3, 4)
    X = triangulate_dlt(P1, P2, pts1, pts2)             # (N, 3)
    depth1 = X[:, 2]                                    # (N,) z in camera 1 = world frame
    depth2 = (X @ R.T + t)[:, 2]                        # (N,) z in camera 2
    return int(((depth1 > 0) & (depth2 > 0)).sum()), X


def recover_pose(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                 K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the one physically possible pose out of the four, by cheirality.

    Returns:
        R (3, 3), t (3,) with ``||t|| = 1``, and X (N, 3) triangulated in camera 1's frame.
    """
    best = (-1, np.eye(3), np.zeros(3), None)
    for R, t in decompose_essential(E):
        count, X = _cheirality_count(R, t, K, pts1, pts2)
        if count > best[0]:
            best = (count, R, t, X)
    _, R, t, X = best
    return R, t, X


@dataclass
class SfMResult:
    """Poses, structure and the bookkeeping that says which of each you may trust."""

    Rs: np.ndarray            # (V, 3, 3) world -> camera rotations
    ts: np.ndarray            # (V, 3)
    Xs: np.ndarray            # (N, 3) world points, rows of invalid tracks are NaN
    registered: np.ndarray    # (V,) bool, which views got a pose
    point_valid: np.ndarray   # (N,) bool, which tracks got a 3-D point
    rms_reprojection: float   # over all valid observations, in pixels


def two_view_initialise(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray,
                        threshold: float = 1.0, rng=None):
    """Bootstrap the reconstruction from one image pair.

    Returns:
        R (3, 3), t (3,), X (N, 3), inliers (N,) bool.  Scale is fixed by ``||t|| = 1``.
    """
    res = ransac_fundamental(pts1, pts2, threshold=threshold, max_iters=3000, rng=rng)
    E = essential_from_fundamental(res.model, K, K)      # (3, 3)
    inl = res.inliers                                    # (N,) bool
    R, t, X_inl = recover_pose(E, pts1[inl], pts2[inl], K)
    X = np.full((len(pts1), 3), np.nan)                  # (N, 3)
    X[inl] = X_inl
    return R, t, X, inl


def _triangulate_track(Ps: list[np.ndarray], pix: np.ndarray, max_error: float) -> np.ndarray | None:
    """Multi-view DLT for one track, rejected if it reprojects badly or sits behind a camera."""
    X = triangulate_multiview(Ps, pix)                    # (3,)
    for P, p in zip(Ps, pix):
        if reprojection_error(P, X[None], p[None])[0] > max_error:
            return None
        if (P[2] @ np.append(X, 1.0)) <= 0:               # depth = third row of P times X
            return None
    return X


def incremental_sfm(
    tracks: np.ndarray,
    visible: np.ndarray,
    K: np.ndarray,
    init_pair: tuple[int, int] | None = None,
    pnp_threshold: float = 3.0,
    triangulation_error: float = 4.0,
    run_bundle_adjustment: bool = True,
    rng=None,
) -> SfMResult:
    """Incremental SfM over precomputed tracks.

    Args:
        tracks: (V, N, 2) pixel location of track ``n`` in view ``v`` (ignored where unseen).
        visible: (V, N) bool, whether view ``v`` observes track ``n``.
        K: (3, 3) shared intrinsics.
        init_pair: views to bootstrap from; defaults to the pair sharing the most tracks.
        triangulation_error: reject a new 3-D point above this reprojection error, in pixels.

    Returns:
        ``SfMResult``.  The reconstruction is in the frame of the first init view and,
        without external constraints, is correct only up to scale.
    """
    V, N = visible.shape
    rng = np.random.default_rng(0) if rng is None else rng
    i, j = init_pair if init_pair else _best_initial_pair(visible)
    both = visible[i] & visible[j]                        # (N,) tracks seen by both
    R_rel, t_rel, X_pair, inl = two_view_initialise(tracks[i][both], tracks[j][both], K, rng=rng)

    Rs = np.repeat(np.eye(3)[None], V, axis=0)            # (V, 3, 3)
    ts = np.zeros((V, 3))                                 # (V, 3)
    Rs[j], ts[j] = R_rel, t_rel
    registered = np.zeros(V, bool)
    registered[[i, j]] = True
    Xs = np.full((N, 3), np.nan)                          # (N, 3)
    Xs[np.flatnonzero(both)[inl]] = X_pair[inl]
    point_valid = np.isfinite(Xs).all(axis=1)             # (N,) bool

    for v in _registration_order(visible, registered, point_valid):
        usable = visible[v] & point_valid                 # (N,) 3-D points this view can see
        if usable.sum() < 6:
            continue
        res = ransac_pnp(Xs[usable], tracks[v][usable], K,
                         threshold=pnp_threshold, max_iters=1500, rng=rng)
        if res.model is None or res.inliers.sum() < 6:
            continue
        Rs[v], ts[v] = res.model
        registered[v] = True
        _grow_structure(tracks, visible, K, Rs, ts, registered, Xs, point_valid,
                        triangulation_error)
        point_valid[:] = np.isfinite(Xs).all(axis=1)

    prob = _assemble_ba_problem(tracks, visible, K, Rs, ts, Xs, registered, point_valid)
    if run_bundle_adjustment and prob is not None:
        ba = bundle_adjust(prob, max_iters=40, huber_delta=3.0, fixed_cameras=(int(i),))
        for k, v in enumerate(np.flatnonzero(registered)):
            Rs[v], ts[v] = ba.Rs[k], ba.ts[k]
        Xs[point_valid] = ba.Xs
        prob = _assemble_ba_problem(tracks, visible, K, Rs, ts, Xs, registered, point_valid)
    rms = 0.0
    if prob is not None:
        from .bundle_adjustment import reprojection_residuals
        r, _ = reprojection_residuals(prob)
        rms = float(np.sqrt((r**2).sum(axis=1).mean()))
    return SfMResult(Rs, ts, Xs, registered, point_valid, rms)


def _best_initial_pair(visible: np.ndarray) -> tuple[int, int]:
    """The view pair sharing the most tracks.

    Production pipelines also demand a wide baseline (a homography must *not* explain the
    matches), because a pair that shares everything may share it from almost the same
    viewpoint, which triangulates to garbage.
    """
    shared = visible.astype(int) @ visible.astype(int).T  # (V, V) common-track counts
    np.fill_diagonal(shared, -1)
    i, j = np.unravel_index(np.argmax(shared), shared.shape)
    return int(i), int(j)


def _registration_order(visible: np.ndarray, registered: np.ndarray,
                        point_valid: np.ndarray) -> list[int]:
    """Remaining views, most-connected first: register where the pose is best constrained."""
    counts = (visible & point_valid).sum(axis=1)          # (V,)
    todo = [v for v in range(len(registered)) if not registered[v]]
    return sorted(todo, key=lambda v: -counts[v])


def _grow_structure(tracks, visible, K, Rs, ts, registered, Xs, point_valid, max_error) -> None:
    """Triangulate tracks that two or more registered views now see, in place."""
    reg = np.flatnonzero(registered)
    Ps = {int(v): projection_matrix(K, Rs[v], ts[v]) for v in reg}  # (3, 4) each
    for n in np.flatnonzero(~point_valid):
        obs = [v for v in reg if visible[v, n]]
        if len(obs) < 2:
            continue
        X = _triangulate_track([Ps[int(v)] for v in obs],
                               np.stack([tracks[v, n] for v in obs]), max_error)
        if X is not None:
            Xs[n] = X


def _assemble_ba_problem(tracks, visible, K, Rs, ts, Xs, registered, point_valid):
    """Pack the registered views and valid points into a :class:`BAProblem`, or ``None``."""
    reg = np.flatnonzero(registered)
    pts = np.flatnonzero(point_valid)
    if len(reg) < 2 or len(pts) < 4:
        return None
    cam_slot = {int(v): k for k, v in enumerate(reg)}
    pt_slot = {int(n): k for k, n in enumerate(pts)}
    cam_idx, pt_idx, pixels = [], [], []
    for v in reg:
        for n in np.flatnonzero(visible[v] & point_valid):
            cam_idx.append(cam_slot[int(v)])
            pt_idx.append(pt_slot[int(n)])
            pixels.append(tracks[v, n])
    return BAProblem(Rs[reg].copy(), ts[reg].copy(), np.repeat(K[None], len(reg), axis=0),
                     Xs[pts].copy(), np.array(cam_idx), np.array(pt_idx), np.array(pixels))
