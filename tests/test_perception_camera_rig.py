import numpy as np

from mlbook.perception import camera_rig as cr


def test_intrinsics_from_fov_focal_length():
    K = cr.intrinsics_from_fov((256, 448), 90.0)
    assert np.isclose(K[0, 0], 224.0)  # tan(45°) = 1 → f = W/2
    assert np.isclose(K[0, 2], 224.0) and np.isclose(K[1, 2], 128.0)


def test_se3_inverse_is_inverse():
    T = cr.camera_pose_from_yaw(0.7, np.array([1.0, -2.0, 1.5]))
    assert np.allclose(T @ cr.se3_inverse(T), np.eye(4), atol=1e-12)


def test_project_unproject_round_trip_on_surround_rig():
    cams = cr.make_surround_rig(n_cams=6, fov_x_deg=70.0, image_hw=(256, 448))
    rng = np.random.default_rng(0)
    pts = rng.uniform(-30, 30, size=(500, 3))
    pts[:, 2] = rng.uniform(-1, 3, size=500)
    seen = 0
    for cam in cams:
        pix, depth, valid = cam.project(pts)
        back = cam.unproject(pix[valid], depth[valid])
        assert np.allclose(back, pts[valid], atol=1e-8)
        seen += valid.sum()
    assert seen > 100  # the rig covers most of the plane


def test_front_camera_sees_forward_points_at_image_centre():
    cam = cr.make_surround_rig(n_cams=6, image_hw=(256, 448))[0]
    pix, depth, valid = cam.project(np.array([[11.0, 0.0, 1.6]]))  # on the optical axis, 10 m ahead of cam0 at x=1
    assert valid[0] and np.isclose(depth[0], 10.0)
    assert np.allclose(pix[0], [224.0, 128.0])
    # A point to the LEFT (+y) must land on the LEFT of the image (smaller u).
    pix_left, _, _ = cam.project(np.array([[11.0, 2.0, 1.6]]))
    assert pix_left[0, 0] < 224.0
    # A point ABOVE the camera lands higher in the image (smaller v).
    pix_up, _, _ = cam.project(np.array([[11.0, 0.0, 3.0]]))
    assert pix_up[0, 1] < 128.0


def test_ego_motion_transform_moves_static_point_backwards():
    T = cr.ego_motion_transform(dx=2.0, dy=0.0, dyaw=0.0)
    p_prev = np.array([10.0, 0.0, 0.0, 1.0])
    p_curr = T @ p_prev
    assert np.allclose(p_curr[:3], [8.0, 0.0, 0.0])  # ego drove 2 m forward → point is 2 m closer
    T_turn = cr.ego_motion_transform(dx=0.0, dy=0.0, dyaw=np.pi / 2)  # ego turned left 90°
    p_curr = T_turn @ np.array([0.0, 5.0, 0.0, 1.0])  # point that was on the left is now straight ahead
    assert np.allclose(p_curr[:3], [5.0, 0.0, 0.0], atol=1e-12)


def test_rolling_shutter_row_time_is_linear():
    t = cr.rolling_shutter_row_time(np.array([0.0, 127.5, 255.0]), 256, 0.03)
    assert np.allclose(t, [0.0, 0.015, 0.03])
