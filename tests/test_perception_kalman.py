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
    F, H = kf.constant_velocity_model(2, dt=1.0)
    f = kf.KalmanFilter(F, H, 0.01 * np.eye(4), 0.5 * np.eye(2), np.zeros(4), np.diag([1.0, 1.0, 100.0, 100.0]))
    rng = np.random.default_rng(0)
    truth = np.array([0.0, 0.0, 1.0, 0.5])
    for _ in range(40):
        truth = F @ truth
        f.predict()
        f.update(truth[:2] + rng.normal(0, 0.5, size=2))
    assert np.allclose(f.x[2:], [1.0, 0.5], atol=0.15)  # velocity recovered from positions only
    assert np.allclose(f.P, f.P.T)
    assert np.all(np.linalg.eigvalsh(f.P) > 0)


def test_mahalanobis_is_chi2_scaled():
    f = kf.KalmanFilter(np.eye(2), np.eye(2), np.zeros((2, 2)), np.eye(2), np.zeros(2), np.eye(2))
    f.predict()
    assert np.isclose(f.mahalanobis(np.array([2.0, 0.0])), 2.0)  # S = 2I → 4/2


def test_covariance_ellipse_axes():
    pts = kf.covariance_ellipse(np.diag([4.0, 1.0]), n_std=1.0)
    assert np.isclose(np.abs(pts[:, 0]).max(), 2.0) and np.isclose(np.abs(pts[:, 1]).max(), 1.0, atol=1e-2)
