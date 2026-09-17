import numpy as np

from mlbook.perception import kalman as kf


def test_constant_velocity_model_matrices():
    F, H = kf.constant_velocity_model(2, dt=0.5)
    assert np.allclose(F @ np.array([0.0, 0.0, 2.0, -4.0]), [1.0, -2.0, 2.0, -4.0])
    assert H.shape == (2, 4) and np.allclose(H @ np.array([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0])


def test_kalman_predict_update_hand_computed_1d():
    # Scalar constant-position model: F = H = 1, Q = 0, R = 1, x0 = 0, P0 = 1.
    f = kf.KalmanFilter(np.eye(1), np.eye(1), np.zeros((1, 1)), np.eye(1), np.zeros(1), np.eye(1))
    f.predict()
    assert np.isclose(f.P[0, 0], 1.0)
    f.update(np.array([2.0]))  # K = 1/(1+1) = 0.5 → x = 1, P = 0.5
    assert np.isclose(f.x[0], 1.0) and np.isclose(f.P[0, 0], 0.5)
    f.predict()
    f.update(np.array([2.0]))  # K = 0.5/1.5 = 1/3 → x = 1 + (2−1)/3 = 4/3, P = 1/3
    assert np.isclose(f.x[0], 4.0 / 3.0) and np.isclose(f.P[0, 0], 1.0 / 3.0)


def test_kalman_tracks_constant_velocity_and_covariance_stays_symmetric():
    # The filter's noise model must MATCH the generator, or the estimate is biased and loose:
    # measurements carry sigma = 0.5, so R = sigma^2 I = 0.25 I (R is a covariance, not a std),
    # and a truly constant velocity means almost no process noise on the velocity states.
    F, H = kf.constant_velocity_model(2, dt=1.0)
    Q = np.diag([0.01, 0.01, 1e-4, 1e-4])  # (4, 4) position jitter only
    R = 0.25 * np.eye(2)  # (2, 2) = sigma^2 of the simulated measurement noise
    truth0 = np.array([0.0, 0.0, 1.0, 0.5])
    for seed in range(5):  # the assertion must hold for any noise realisation, not one lucky seed
        rng = np.random.default_rng(seed)
        f = kf.KalmanFilter(F, H, Q, R, np.zeros(4), np.diag([1.0, 1.0, 100.0, 100.0]))
        truth = truth0.copy()
        for _ in range(40):
            truth = F @ truth
            f.predict()
            f.update(truth[:2] + rng.normal(0, 0.5, size=2))
        assert np.allclose(f.x[2:], [1.0, 0.5], atol=0.05)  # velocity recovered from positions only
        assert np.allclose(f.x[:2], truth[:2], atol=0.5)
        assert np.allclose(f.P, f.P.T)  # the Joseph form keeps P symmetric
        assert np.all(np.linalg.eigvalsh(f.P) > 0)
        assert f.P[2, 2] < 1e-2  # velocity uncertainty collapsed far below its P0 = 100


def test_mahalanobis_is_chi2_scaled():
    f = kf.KalmanFilter(np.eye(2), np.eye(2), np.zeros((2, 2)), np.eye(2), np.zeros(2), np.eye(2))
    f.predict()
    assert np.isclose(f.mahalanobis(np.array([2.0, 0.0])), 2.0)  # S = 2I → 4/2


def test_covariance_ellipse_axes():
    P = np.diag([4.0, 1.0])
    pts = kf.covariance_ellipse(P, n_std=1.0, n_points=64)
    assert pts.shape == (65, 2) and np.allclose(pts[0], pts[-1])  # closed polygon
    # The semi-axes are sqrt of the eigenvalues; the sampling hits them exactly.
    assert np.isclose(np.abs(pts[:, 0]).max(), 2.0)
    assert np.isclose(np.abs(pts[:, 1]).max(), 1.0)
    # Every point lies on the level set p^T P^-1 p = n_std^2 — the defining property.
    quad = np.einsum("ij,jk,ik->i", pts, np.linalg.inv(P), pts)
    assert np.allclose(quad, 1.0)
    # A rotated covariance gives a rotated ellipse with the same quadratic form.
    Rot = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    P_rot = Rot @ P @ Rot.T
    pts_rot = kf.covariance_ellipse(P_rot, n_std=2.0, n_points=64)
    quad_rot = np.einsum("ij,jk,ik->i", pts_rot, np.linalg.inv(P_rot), pts_rot)
    assert np.allclose(quad_rot, 4.0)
