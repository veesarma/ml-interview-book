import numpy as np

from mlbook.geometry import camera as C
from mlbook.geometry import epipolar as E


def _two_views():
    K = C.intrinsics(600.0, 600.0, 320.0, 240.0)
    R1, t1 = C.look_at(np.array([0.0, -4.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    R2, t2 = C.look_at(np.array([1.5, -3.5, 1.3]), np.array([0.0, 0.0, 1.0]))
    X = np.random.uniform(-1, 1, size=(40, 3)) + np.array([0.0, 0.0, 1.0])
    p1, _ = C.project(X, K, R1, t1)
    p2, _ = C.project(X, K, R2, t2)
    return K, (R1, t1), (R2, t2), X, p1, p2


def test_essential_from_pose_satisfies_constraint():
    K, (R1, t1), (R2, t2), X, p1, p2 = _two_views()
    R = R2 @ R1.T  # relative rotation cam1 → cam2
    t = t2 - R @ t1
    Ess = E.essential_from_pose(R, t)
    F = E.fundamental_from_essential(Ess, K, K)
    x1 = C.to_homogeneous(p1)
    x2 = C.to_homogeneous(p2)
    assert np.allclose(np.sum(x2 * (x1 @ F.T), axis=1), 0.0, atol=1e-9)


def test_normalise_points_statistics():
    pts = np.random.uniform(0, 500, size=(50, 2))
    xn, T = E.normalise_points(pts)
    assert np.allclose(xn[:, :2].mean(axis=0), 0.0, atol=1e-10)
    assert abs(np.linalg.norm(xn[:, :2], axis=1).mean() - np.sqrt(2)) < 1e-10


def test_eight_point_recovers_fundamental_matrix():
    K, (R1, t1), (R2, t2), X, p1, p2 = _two_views()
    F_est = E.eight_point(p1, p2)
    assert np.linalg.matrix_rank(F_est, tol=1e-8) == 2
    assert np.all(E.sampson_distance(F_est, p1, p2) < 1e-6)
    R = R2 @ R1.T
    t = t2 - R @ t1
    F_true = E.fundamental_from_essential(E.essential_from_pose(R, t), K, K)
    F_true /= np.linalg.norm(F_true)
    sign = np.sign(np.sum(F_true * F_est))
    assert np.allclose(sign * F_est, F_true, atol=1e-6)


def test_epipolar_lines_pass_through_matches_and_epipole():
    K, (R1, t1), (R2, t2), X, p1, p2 = _two_views()
    F = E.eight_point(p1, p2)
    lines = E.epipolar_lines(F, p1)  # (N, 3)
    dist = np.abs(np.sum(lines * C.to_homogeneous(p2), axis=1)) / np.linalg.norm(lines[:, :2], axis=1)
    assert np.all(dist < 1e-5)
    e1, e2 = E.epipoles(F)
    # the epipole in image 2 is the projection of camera-1's centre
    centre1 = -R1.T @ t1
    proj, _ = C.project(centre1[None], K, R2, t2)
    assert np.allclose(e2, proj[0], atol=1e-3)
