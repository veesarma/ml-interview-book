"""Incremental SfM: essential-matrix decomposition, cheirality, and an end-to-end
reconstruction of a synthetic aerial survey recovered up to a similarity."""

import numpy as np

from mlbook.geometry import camera as C
from mlbook.geometry import icp as I
from mlbook.geometry import sfm as S


def _survey(n_views: int = 6, n_pts: int = 120, seed: int = 0):
    """Overlapping nadir passes over a patch of terrain, as an aerial survey flies it."""
    rng = np.random.default_rng(seed)
    K = C.intrinsics(900.0, 900.0, 640.0, 480.0)  # (3, 3)
    Rs, ts = [], []
    for v in range(n_views):
        eye = np.array([-25.0 + 10.0 * v, -6.0 + 3.0 * (v % 2), 55.0 + rng.normal(0, 0.5)])
        R, t = C.look_at(eye, np.array([-25.0 + 10.0 * v, 0.0, 0.0]))
        Rs.append(R)
        ts.append(t)
    Rs, ts = np.stack(Rs), np.stack(ts)                 # (V, 3, 3), (V, 3)
    Xs = np.column_stack([                               # (N, 3)
        rng.uniform(-35, 35, n_pts), rng.uniform(-16, 16, n_pts), rng.uniform(0, 5, n_pts)
    ])
    tracks = np.zeros((n_views, n_pts, 2))               # (V, N, 2)
    visible = np.zeros((n_views, n_pts), bool)           # (V, N)
    for v in range(n_views):
        uv, depth = C.project(Xs, K, Rs[v], ts[v])       # (N, 2), (N,)
        tracks[v] = uv + rng.normal(0, 0.25, uv.shape)   # sub-pixel detector noise
        visible[v] = (depth > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 1280) \
            & (uv[:, 1] > 0) & (uv[:, 1] < 960)
    return K, Rs, ts, Xs, tracks, visible


def test_essential_from_fundamental_lands_on_the_essential_manifold():
    K, Rs, ts, Xs, tracks, visible = _survey(seed=1)
    from mlbook.geometry.epipolar import eight_point
    both = visible[0] & visible[1]
    F = eight_point(tracks[0][both], tracks[1][both])
    E = S.essential_from_fundamental(F, K, K)
    sv = np.linalg.svd(E, compute_uv=False)              # (3,)
    assert np.isclose(sv[0], sv[1]) and np.isclose(sv[2], 0.0, atol=1e-12)


def test_decompose_essential_gives_four_poses_and_cheirality_picks_one():
    K, Rs, ts, Xs, _, _ = _survey(seed=2)
    # relative pose from view 0 to view 1, which is what E encodes
    R_rel = Rs[1] @ Rs[0].T                              # (3, 3)
    t_rel = ts[1] - R_rel @ ts[0]                        # (3,)
    E = C.skew_symmetric(t_rel / np.linalg.norm(t_rel)) @ R_rel

    candidates = S.decompose_essential(E)
    assert len(candidates) == 4
    assert all(np.isclose(np.linalg.det(R), 1.0) for R, _ in candidates)
    assert sum(np.allclose(R, R_rel, atol=1e-6) for R, _ in candidates) == 2  # R and its twist

    # project into view 0's frame and let cheirality choose
    X0 = Xs @ Rs[0].T + ts[0]                            # (N, 3) points in camera 0
    p0, _ = C.project(X0, K, np.eye(3), np.zeros(3))
    p1, _ = C.project(X0, K, R_rel, t_rel / np.linalg.norm(t_rel))
    R_hat, t_hat, X_hat = S.recover_pose(E, p0, p1, K)
    assert np.allclose(R_hat, R_rel, atol=1e-6)
    assert np.allclose(t_hat, t_rel / np.linalg.norm(t_rel), atol=1e-6)
    assert (X_hat[:, 2] > 0).all()                       # every point in front of camera 0


def test_two_view_initialise_recovers_rotation_and_baseline_direction():
    K, Rs, ts, Xs, tracks, visible = _survey(seed=3)
    both = visible[0] & visible[1]
    R_hat, t_hat, X, inl = S.two_view_initialise(tracks[0][both], tracks[1][both], K,
                                                 rng=np.random.default_rng(4))
    R_rel = Rs[1] @ Rs[0].T
    t_rel = ts[1] - R_rel @ ts[0]

    assert inl.mean() > 0.9
    assert np.allclose(R_hat, R_rel, atol=2e-2)
    # translation is recovered only in direction: two views cannot see absolute scale
    assert np.isclose(np.linalg.norm(t_hat), 1.0)
    cos = t_hat @ (t_rel / np.linalg.norm(t_rel))
    angle_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    # a 10 m baseline at 55 m altitude is a narrow-baseline pair, so the direction is
    # the weakly conditioned part of the estimate: a few degrees off at 0.25 px noise.
    # Bundle adjustment over the whole strip is what pulls this back down.
    assert angle_deg < 5.0


def test_incremental_sfm_reconstructs_the_survey_up_to_a_similarity():
    K, Rs, ts, Xs, tracks, visible = _survey(n_views=6, n_pts=140, seed=5)

    res = S.incremental_sfm(tracks, visible, K, rng=np.random.default_rng(6))

    assert res.registered.all()                      # every view found a pose
    assert res.point_valid.mean() > 0.8
    assert res.rms_reprojection < 1.0                # at the feature-noise floor

    # the reconstruction is metric only up to a similarity, so align before comparing
    good = res.point_valid
    R_a, t_a, s_a = I.kabsch_umeyama(res.Xs[good], Xs[good], with_scale=True)
    aligned = I.apply_similarity(res.Xs[good], R_a, t_a, s_a)   # (P, 3)
    assert np.linalg.norm(aligned - Xs[good], axis=1).mean() < 0.15

    # camera centres line up under the same similarity
    centres_hat = np.stack([-res.Rs[v].T @ res.ts[v] for v in range(6)])   # (6, 3)
    centres_gt = np.stack([-Rs[v].T @ ts[v] for v in range(6)])            # (6, 3)
    assert np.linalg.norm(I.apply_similarity(centres_hat, R_a, t_a, s_a) - centres_gt,
                          axis=1).max() < 0.5


def test_sfm_without_bundle_adjustment_is_measurably_worse():
    K, Rs, ts, Xs, tracks, visible = _survey(n_views=5, n_pts=110, seed=7)
    with_ba = S.incremental_sfm(tracks, visible, K, rng=np.random.default_rng(8))
    without = S.incremental_sfm(tracks, visible, K, run_bundle_adjustment=False,
                                rng=np.random.default_rng(8))
    assert with_ba.rms_reprojection < without.rms_reprojection
