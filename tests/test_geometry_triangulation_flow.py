import numpy as np

from mlbook.geometry import camera as C
from mlbook.geometry import optical_flow as OF
from mlbook.geometry import triangulation as T
from mlbook.vision.image_ops import gaussian_blur


def _views():
    K = C.intrinsics(600.0, 600.0, 320.0, 240.0)
    R1, t1 = C.look_at(np.array([0.0, -4.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    R2, t2 = C.look_at(np.array([2.0, -3.5, 1.5]), np.array([0.0, 0.0, 1.0]))
    X = np.random.uniform(-1, 1, size=(25, 3)) + np.array([0.0, 0.0, 1.0])
    return K, (R1, t1), (R2, t2), X


def test_triangulate_dlt_recovers_points():
    K, (R1, t1), (R2, t2), X = _views()
    P1, P2 = C.projection_matrix(K, R1, t1), C.projection_matrix(K, R2, t2)
    p1, _ = C.project(X, K, R1, t1)
    p2, _ = C.project(X, K, R2, t2)
    X_hat = T.triangulate_dlt(P1, P2, p1, p2)
    assert np.allclose(X_hat, X, atol=1e-6)
    noisy = T.triangulate_dlt(P1, P2, p1 + np.random.randn(*p1.shape) * 0.5, p2 + np.random.randn(*p2.shape) * 0.5)
    assert np.linalg.norm(noisy - X, axis=1).mean() < 0.05
    assert T.reprojection_error(P1, X_hat, p1).max() < 1e-4


def test_triangulate_multiview():
    K, (R1, t1), (R2, t2), X = _views()
    R3, t3 = C.look_at(np.array([-2.0, -3.0, 0.5]), np.array([0.0, 0.0, 1.0]))
    Ps = [C.projection_matrix(K, R, t) for R, t in [(R1, t1), (R2, t2), (R3, t3)]]
    pix = np.stack([C.project(X[:1], K, R, t)[0][0] for R, t in [(R1, t1), (R2, t2), (R3, t3)]])  # (3, 2)
    assert np.allclose(T.triangulate_multiview(Ps, pix), X[0], atol=1e-6)


def test_pnp_dlt_recovers_pose():
    K, (R1, t1), _, X = _views()
    p1, _ = C.project(X, K, R1, t1)
    R_hat, t_hat = T.pnp_dlt(X, p1, K)
    assert np.allclose(R_hat, R1, atol=1e-6) and np.allclose(t_hat, t1, atol=1e-6)


def test_lucas_kanade_recovers_subpixel_shift():
    rng = np.random.default_rng(0)
    base = gaussian_blur(rng.random((64, 64)), 2.0)  # smooth texture: LK's linearisation holds
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    from mlbook.vision.image_ops import bilinear_sample
    dy, dx = 1.3, -0.7
    shifted = bilinear_sample(base, (yy - dy).ravel(), (xx - dx).ravel()).reshape(64, 64)  # I2(x) = I1(x − d)
    pts = np.array([[20.0, 20.0], [32.0, 40.0], [45.0, 25.0]])
    flow, min_eig = OF.lucas_kanade(base, shifted, pts, window=9, iters=8)
    assert np.all(min_eig > 0)
    assert np.allclose(flow, [[dy, dx]] * 3, atol=0.1)  # sub-pixel: residual error from bilinear warp + finite differences


def test_image_gradients_on_ramp():
    yy, xx = np.mgrid[0:8, 0:8].astype(float)
    Ix, Iy = OF.image_gradients(2 * xx + 3 * yy)
    assert np.allclose(Ix[1:-1, 1:-1], 2.0) and np.allclose(Iy[1:-1, 1:-1], 3.0)
