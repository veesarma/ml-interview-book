# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/sfm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k sfm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/sfm --force

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
    raise NotImplementedError('TODO: implement essential_from_fundamental (see the reference in src/mlbook)')

def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four ``(R, t)`` candidates consistent with ``E = [t]_x R``.

    With ``E = U diag(1,1,0) V^T`` and ``W`` the 90-degree rotation about z, the candidates
    are ``(U W V^T, +u_3)``, ``(U W V^T, -u_3)``, ``(U W^T V^T, +u_3)``, ``(U W^T V^T, -u_3)``.
    They correspond to the true solution plus a twist of one camera by 180 degrees about the
    baseline and a sign flip of the baseline, both of which put points behind a camera.

    Returns:
        list of 4 ``(R (3,3), t (3,))`` with ``||t|| = 1``.
    """
    raise NotImplementedError('TODO: implement decompose_essential (see the reference in src/mlbook)')

def _cheirality_count(R: np.ndarray, t: np.ndarray, K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> tuple[int, np.ndarray]:
    """How many triangulated points sit in front of *both* cameras, and where they are."""
    raise NotImplementedError('TODO: implement _cheirality_count (see the reference in src/mlbook)')

def recover_pose(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the one physically possible pose out of the four, by cheirality.

    Returns:
        R (3, 3), t (3,) with ``||t|| = 1``, and X (N, 3) triangulated in camera 1's frame.
    """
    raise NotImplementedError('TODO: implement recover_pose (see the reference in src/mlbook)')

@dataclass
class SfMResult:
    """Poses, structure and the bookkeeping that says which of each you may trust."""
    Rs: np.ndarray
    ts: np.ndarray
    Xs: np.ndarray
    registered: np.ndarray
    point_valid: np.ndarray
    rms_reprojection: float

def two_view_initialise(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, threshold: float=1.0, rng=None):
    """Bootstrap the reconstruction from one image pair.

    Returns:
        R (3, 3), t (3,), X (N, 3), inliers (N,) bool.  Scale is fixed by ``||t|| = 1``.
    """
    raise NotImplementedError('TODO: implement two_view_initialise (see the reference in src/mlbook)')

def _triangulate_track(Ps: list[np.ndarray], pix: np.ndarray, max_error: float) -> np.ndarray | None:
    """Multi-view DLT for one track, rejected if it reprojects badly or sits behind a camera."""
    raise NotImplementedError('TODO: implement _triangulate_track (see the reference in src/mlbook)')

def incremental_sfm(tracks: np.ndarray, visible: np.ndarray, K: np.ndarray, init_pair: tuple[int, int] | None=None, pnp_threshold: float=3.0, triangulation_error: float=4.0, run_bundle_adjustment: bool=True, rng=None) -> SfMResult:
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
    raise NotImplementedError('TODO: implement incremental_sfm (see the reference in src/mlbook)')

def _best_initial_pair(visible: np.ndarray) -> tuple[int, int]:
    """The view pair sharing the most tracks.

    Production pipelines also demand a wide baseline (a homography must *not* explain the
    matches), because a pair that shares everything may share it from almost the same
    viewpoint, which triangulates to garbage.
    """
    raise NotImplementedError('TODO: implement _best_initial_pair (see the reference in src/mlbook)')

def _registration_order(visible: np.ndarray, registered: np.ndarray, point_valid: np.ndarray) -> list[int]:
    """Remaining views, most-connected first: register where the pose is best constrained."""
    raise NotImplementedError('TODO: implement _registration_order (see the reference in src/mlbook)')

def _grow_structure(tracks, visible, K, Rs, ts, registered, Xs, point_valid, max_error) -> None:
    """Triangulate tracks that two or more registered views now see, in place."""
    raise NotImplementedError('TODO: implement _grow_structure (see the reference in src/mlbook)')

def _assemble_ba_problem(tracks, visible, K, Rs, ts, Xs, registered, point_valid):
    """Pack the registered views and valid points into a :class:`BAProblem`, or ``None``."""
    raise NotImplementedError('TODO: implement _assemble_ba_problem (see the reference in src/mlbook)')
