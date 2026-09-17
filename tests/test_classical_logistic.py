"""Tests for logistic_regression.py and softmax_regression.py — one focused test per retype target."""
import numpy as np

from mlbook.classical import logistic_regression as lg
from mlbook.classical import softmax_regression as sm


def _binary(N=300, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))
    X = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, d+1) with bias
    w_true = rng.standard_normal(d + 1)
    y = (rng.random(N) < lg.sigmoid(X @ w_true)).astype(float)
    return X, y, w_true


def test_sigmoid_stable_and_correct():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    p = lg.sigmoid(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p[1:4], 1 / (1 + np.exp(-z[1:4])))
    assert p[0] == 0.0 and p[-1] == 1.0
    np.testing.assert_allclose(lg.log_sigmoid(z[1:4]), np.log(p[1:4]))


def test_bce_loss_known_value():
    X = np.array([[1.0], [1.0]])
    y = np.array([1.0, 0.0])
    w = np.array([0.0])  # p = 0.5 for both -> loss = log 2
    np.testing.assert_allclose(lg.bce_loss(X, y, w), np.log(2.0))


def test_bce_gradient_finite_difference():
    X, y, _ = _binary(N=40)
    w = np.random.default_rng(3).standard_normal(X.shape[1])
    sw = np.random.default_rng(4).random(len(y)) + 0.5
    g = lg.bce_gradient(X, y, w, sw, l2=0.1)
    eps = 1e-5
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = eps
        fd = (lg.bce_loss(X, y, w + e, sw, 0.1) - lg.bce_loss(X, y, w - e, sw, 0.1)) / (2 * eps)
        np.testing.assert_allclose(g[j], fd, atol=1e-6)


def test_bce_hessian_finite_difference_and_psd():
    X, y, _ = _binary(N=40)
    w = np.random.default_rng(3).standard_normal(X.shape[1])
    H = lg.bce_hessian(X, w, l2=0.1)
    eps = 1e-5
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = eps
        fd = (lg.bce_gradient(X, y, w + e, l2=0.1) - lg.bce_gradient(X, y, w - e, l2=0.1)) / (2 * eps)
        np.testing.assert_allclose(H[:, j], fd, atol=1e-5)
    assert np.linalg.eigvalsh(H).min() >= -1e-10


def test_fit_logistic_gd_zero_gradient_at_solution():
    X, y, _ = _binary(N=500)
    w = lg.fit_logistic_gd(X, y, lr=0.5, n_steps=4000, l2=1e-3)
    assert np.abs(lg.bce_gradient(X, y, w, l2=1e-3)).max() < 1e-4


def test_fit_logistic_newton_matches_gd_and_recovers_weights():
    X, y, w_true = _binary(N=2000)
    w_newton = lg.fit_logistic_newton(X, y, l2=1e-4)
    w_gd = lg.fit_logistic_gd(X, y, lr=0.5, n_steps=3000, l2=1e-4)
    np.testing.assert_allclose(w_newton, w_gd, atol=1e-3)
    assert np.corrcoef(w_newton, w_true)[0, 1] > 0.95
    assert np.abs(lg.bce_gradient(X, y, w_newton, l2=1e-4)).max() < 1e-8


def test_platt_scaling_recovers_affine_map():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(5000)
    y = (rng.random(5000) < lg.sigmoid(2.0 * s - 0.5)).astype(float)
    a, b = lg.platt_scaling(s, y)
    assert abs(a - 2.0) < 0.2 and abs(b + 0.5) < 0.15


def test_balanced_class_weights_equalise_classes():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
    w = lg.balanced_class_weights(y)
    assert abs(w[y == 1].sum() - w[y == 0].sum()) < 1e-12
    np.testing.assert_allclose(w.sum(), len(y))


def test_one_vs_rest_fit_predict():
    rng = np.random.default_rng(0)
    N, K = 600, 3
    y = rng.integers(0, K, N)
    X = np.array([[2, 0], [-2, 0], [0, 3.0]])[y] + 0.6 * rng.standard_normal((N, 2))
    Xb = np.concatenate([X, np.ones((N, 1))], axis=1)
    W = lg.one_vs_rest_fit(Xb, y, K, l2=1e-3)
    assert W.shape == (3, K)
    assert (lg.one_vs_rest_predict(Xb, W) == y).mean() > 0.95


# ------------------------------- softmax ---------------------------------- #
def test_log_softmax_and_softmax_stable():
    Z = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    P = sm.softmax(Z)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert np.all(np.isfinite(sm.log_softmax(Z)))
    np.testing.assert_allclose(P[1], np.exp(Z[1]) / np.exp(Z[1]).sum())
    # temperature: T -> large flattens
    assert sm.softmax(Z[1:], temperature=100.0).std() < P[1].std()


def test_softmax_jacobian_finite_difference():
    z = np.array([0.3, -1.2, 2.0, 0.1])
    p = sm.softmax(z[None, :])[0]
    J = sm.softmax_jacobian(p)
    eps = 1e-6
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        col = (sm.softmax((z + e)[None, :])[0] - sm.softmax((z - e)[None, :])[0]) / (2 * eps)
        np.testing.assert_allclose(J[:, j], col, atol=1e-7)


def test_cross_entropy_known_value_and_smoothing():
    Z = np.array([[0.0, 0.0, 0.0, 0.0]])
    Y = sm.one_hot(np.array([2]), 4)
    np.testing.assert_allclose(sm.cross_entropy(Z, Y), np.log(4.0))
    Ys = sm.smooth_labels(Y, 0.2)
    np.testing.assert_allclose(Ys[0], [0.05, 0.05, 0.85, 0.05])
    np.testing.assert_allclose(Ys.sum(axis=1), 1.0)


def test_cross_entropy_grad_logits_is_p_minus_y():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((6, 4))
    Y = sm.smooth_labels(sm.one_hot(rng.integers(0, 4, 6), 4), eps=0.1)
    for T in [1.0, 2.5]:
        G = sm.cross_entropy_grad_logits(Z, Y, temperature=T)
        np.testing.assert_allclose(G, (sm.softmax(Z, T) - Y) / (T * 6))
        eps = 1e-6
        for i in range(6):
            for k in range(4):
                E = np.zeros_like(Z)
                E[i, k] = eps
                fd = (sm.cross_entropy(Z + E, Y, T) - sm.cross_entropy(Z - E, Y, T)) / (2 * eps)
                np.testing.assert_allclose(G[i, k], fd, atol=1e-6)


def test_softmax_regression_grad_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3))
    Y = sm.one_hot(rng.integers(0, 4, 8), 4)
    W = rng.standard_normal((3, 4))
    G = sm.softmax_regression_grad(X, Y, W, l2=0.05)
    eps = 1e-6
    for i in range(3):
        for k in range(4):
            E = np.zeros_like(W)
            E[i, k] = eps
            fd = (sm.softmax_regression_loss(X, Y, W + E, 0.05) - sm.softmax_regression_loss(X, Y, W - E, 0.05)) / (2 * eps)
            np.testing.assert_allclose(G[i, k], fd, atol=1e-6)


def test_fit_softmax_gd_learns_and_label_smoothing_shrinks_logits():
    rng = np.random.default_rng(0)
    N, d, K = 600, 2, 3
    y = rng.integers(0, K, N)
    X = np.array([[2, 0], [-2, 0], [0, 3.0]])[y] + 0.6 * rng.standard_normal((N, d))
    Xb = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, 3)
    W = sm.fit_softmax_gd(Xb, y, K, lr=0.5, n_steps=1500)
    assert (sm.predict(Xb, W) == y).mean() > 0.95
    W_ls = sm.fit_softmax_gd(Xb, y, K, lr=0.5, n_steps=1500, label_smoothing=0.2)
    assert np.linalg.norm(W_ls) < np.linalg.norm(W)
