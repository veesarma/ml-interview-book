"""Bundle adjustment: exponential map, analytic Jacobians vs finite differences,
convergence from a perturbed start, robustness to a gross outlier, and GCP anchoring."""

import numpy as np

from mlbook.geometry import bundle_adjustment as BA
from mlbook.geometry import camera as C


def _aerial_survey(n_cams: int = 5, n_pts: int = 60, seed: int = 0):
    """A nadir-ish aerial strip: cameras on a line at 60 m, terrain points below."""
    rng = np.random.default_rng(seed)
    K = C.intrinsics(900.0, 900.0, 640.0, 480.0)  # (3, 3)
    Rs, ts = [], []
    for i in range(n_cams):
        eye = np.array([-20.0 + 10.0 * i, rng.normal(0, 1.0), 60.0])  # (3,)
        R, t = C.look_at(eye, np.array([-20.0 + 10.0 * i, 0.0, 0.0]))
        Rs.append(R)
        ts.append(t)
    Rs, ts = np.stack(Rs), np.stack(ts)                    # (C, 3, 3), (C, 3)
    Ks = np.repeat(K[None], n_cams, axis=0)                # (C, 3, 3)
    Xs = np.column_stack([                                  # (P, 3) ground with relief
        rng.uniform(-28, 28, n_pts), rng.uniform(-14, 14, n_pts), rng.uniform(0, 6, n_pts)
    ])
    cam_idx, pt_idx, pixels = [], [], []
    for i in range(n_cams):
        uv, _ = C.project(Xs, Ks[i], Rs[i], ts[i])          # (P, 2)
        visible = (uv[:, 0] > 0) & (uv[:, 0] < 1280) & (uv[:, 1] > 0) & (uv[:, 1] < 960)
        for j in np.flatnonzero(visible):
            cam_idx.append(i)
            pt_idx.append(int(j))
            pixels.append(uv[j])
    prob = BA.BAProblem(Rs, ts, Ks, Xs, np.array(cam_idx), np.array(pt_idx), np.array(pixels))
    return rng, prob


def _rms(prob: BA.BAProblem) -> float:
    r, _ = BA.reprojection_residuals(prob)
    return float(np.sqrt((r**2).sum(axis=1).mean()))


def test_so3_exp_matches_rodrigues_and_is_a_rotation():
    axis, angle = np.array([0.3, -0.5, 0.8]), 0.7
    R = BA.so3_exp(axis / np.linalg.norm(axis) * angle)
    assert np.allclose(R, C.rotation_from_axis_angle(axis, angle))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    # the small-angle branch stays accurate where the 1/theta^2 form would cancel badly
    tiny = np.array([1e-10, -2e-10, 3e-10])
    assert np.allclose(BA.so3_exp(tiny), np.eye(3) + C.skew_symmetric(tiny), atol=1e-18)


def test_analytic_jacobians_match_finite_differences():
    _, prob = _aerial_survey(n_cams=3, n_pts=12, seed=1)
    r0, Y = BA.reprojection_residuals(prob)
    J_cam, J_pt = BA.observation_jacobians(prob, Y)
    eps = 1e-6

    def residual_with(Rs, ts, Xs):
        return BA.reprojection_residuals(
            BA.BAProblem(Rs, ts, prob.Ks, Xs, prob.cam_idx, prob.pt_idx, prob.pixels))[0]

    for axis in range(6):  # 3 rotation + 3 translation dof of camera 1
        d = np.zeros(6)
        d[axis] = eps
        Rs, ts = prob.Rs.copy(), prob.ts.copy()
        dR = BA.so3_exp(d[:3])
        Rs[1], ts[1] = dR @ prob.Rs[1], dR @ prob.ts[1] + d[3:]
        fd = (residual_with(Rs, ts, prob.Xs) - r0) / eps            # (M, 2)
        mask = prob.cam_idx == 1
        assert np.allclose(fd[mask], J_cam[mask, :, axis], atol=1e-3)

    for axis in range(3):  # the 3 dof of point 4
        Xs = prob.Xs.copy()
        Xs[4, axis] += eps
        fd = (residual_with(prob.Rs, prob.ts, Xs) - r0) / eps       # (M, 2)
        mask = prob.pt_idx == 4
        assert np.allclose(fd[mask], J_pt[mask, :, axis], atol=1e-3)


def test_bundle_adjust_recovers_poses_and_structure_from_a_perturbed_start():
    rng, prob = _aerial_survey(n_cams=5, n_pts=60, seed=2)
    prob.pixels += rng.normal(0, 0.3, prob.pixels.shape)  # sub-pixel feature noise

    # perturb every camera but the first, and every point: the state SfM hands to BA
    Rs, ts, Xs = prob.Rs.copy(), prob.ts.copy(), prob.Xs.copy()
    for i in range(1, len(Rs)):
        dR = BA.so3_exp(rng.normal(0, 0.01, 3))
        Rs[i], ts[i] = dR @ Rs[i], dR @ ts[i] + rng.normal(0, 0.4, 3)
    Xs += rng.normal(0, 0.5, Xs.shape)
    start = BA.BAProblem(Rs, ts, prob.Ks, Xs, prob.cam_idx, prob.pt_idx, prob.pixels)
    assert _rms(start) > 5.0

    res = BA.bundle_adjust(start, max_iters=60, huber_delta=3.0)
    refined = BA.BAProblem(res.Rs, res.ts, prob.Ks, res.Xs,
                           prob.cam_idx, prob.pt_idx, prob.pixels)

    assert _rms(refined) < 0.5                      # down to the noise floor
    assert res.costs == sorted(res.costs, reverse=True)  # LM only ever accepts descent
    # camera 0 was the gauge anchor, so it must not have moved at all
    assert np.allclose(res.Rs[0], prob.Rs[0]) and np.allclose(res.ts[0], prob.ts[0])


def test_huber_kernel_caps_the_influence_of_a_gross_mismatch():
    rng, prob = _aerial_survey(n_cams=4, n_pts=50, seed=3)
    clean_Xs = prob.Xs.copy()
    prob.pixels[7] += np.array([180.0, -140.0])   # one catastrophic feature mismatch

    Xs = prob.Xs + rng.normal(0, 0.3, prob.Xs.shape)
    start = BA.BAProblem(prob.Rs.copy(), prob.ts.copy(), prob.Ks, Xs,
                         prob.cam_idx, prob.pt_idx, prob.pixels)

    robust = BA.bundle_adjust(start, max_iters=40, huber_delta=2.0)
    plain = BA.bundle_adjust(start, max_iters=40, huber_delta=1e9)  # no robustification

    def structure_error(res):
        return np.linalg.norm(res.Xs - clean_Xs, axis=1).mean()

    assert structure_error(robust) < structure_error(plain)
    assert BA.huber_weights(np.array([1.0, 100.0]), 2.0)[1] == 0.02  # 2 / 100


def test_fixed_points_act_as_ground_control_and_hold_the_scale():
    rng, prob = _aerial_survey(n_cams=4, n_pts=40, seed=4)
    Xs = prob.Xs + rng.normal(0, 0.4, prob.Xs.shape)
    Xs[[0, 1, 2]] = prob.Xs[[0, 1, 2]]            # three surveyed ground control points
    start = BA.BAProblem(prob.Rs.copy(), prob.ts.copy(), prob.Ks, Xs,
                         prob.cam_idx, prob.pt_idx, prob.pixels)

    res = BA.bundle_adjust(start, max_iters=40, fixed_points=(0, 1, 2))

    assert np.allclose(res.Xs[[0, 1, 2]], prob.Xs[[0, 1, 2]])     # GCPs never moved
    assert np.linalg.norm(res.Xs - prob.Xs, axis=1).mean() < 0.05  # metric structure recovered
