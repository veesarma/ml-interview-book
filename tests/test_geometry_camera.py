import numpy as np

from mlbook.geometry import camera as C


def _scene():
    K = C.intrinsics(500.0, 500.0, 320.0, 240.0)
    R, t = C.look_at(eye=np.array([2.0, -3.0, 1.0]), target=np.array([0.0, 0.0, 0.5]))
    return K, R, t


def test_intrinsics_and_rotation_from_axis_angle():
    K = C.intrinsics(500.0, 400.0, 320.0, 240.0)
    assert K[0, 0] == 500.0 and K[1, 2] == 240.0 and K[2, 2] == 1.0
    R = C.rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12


def test_look_at_is_valid_extrinsic():
    K, R, t = _scene()
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) > 0
    # the target lies on the optical axis: it projects to the principal point
    px, depth = C.project(np.array([[0.0, 0.0, 0.5]]), K, R, t)
    assert np.allclose(px[0], [320.0, 240.0]) and depth[0] > 0


def test_project_hand_computed():
    K = C.intrinsics(100.0, 100.0, 50.0, 50.0)
    R, t = np.eye(3), np.zeros(3)
    px, z = C.project(np.array([[1.0, 2.0, 4.0]]), K, R, t)
    assert np.allclose(px[0], [50 + 100 * 0.25, 50 + 100 * 0.5]) and z[0] == 4.0


def test_project_unproject_roundtrip_with_distortion():
    K, R, t = _scene()
    dist = (-0.2, 0.05, 0.001, -0.001, 0.0)
    P = np.random.uniform(-1, 1, size=(30, 3))  # (N, 3) points near the target
    px, depth = C.project(P, K, R, t, dist)
    assert np.all(depth > 0)
    back = C.unproject(px, depth, K, R, t, dist)
    assert np.allclose(back, P, atol=1e-8)


def test_distort_undistort_inverse():
    dist = (-0.3, 0.1, 0.0, 0.0, 0.0)
    xn = np.random.uniform(-0.5, 0.5, size=(20, 2))
    assert np.allclose(C.undistort(C.distort(xn, dist), dist, iters=20), xn, atol=1e-9)


def test_projection_matrix_homogeneous():
    K, R, t = _scene()
    P = C.projection_matrix(K, R, t)
    X = np.random.uniform(-1, 1, size=(5, 3))
    px_h = C.to_homogeneous(X) @ P.T  # (5, 3)
    px, _ = C.project(X, K, R, t)
    assert np.allclose(C.from_homogeneous(px_h), px)
