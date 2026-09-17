"""Tests for src/mlbook/classical/linear_regression.py — one focused test per retype target."""
import numpy as np

from mlbook.classical import linear_regression as lr


def _data(N=200, d=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))  # (N, d)
    w_true = rng.standard_normal(d)  # (d,)
    y = X @ w_true + noise * rng.standard_normal(N)  # (N,)
    return X, y, w_true


def test_fit_ols_normal_equations_matches_lstsq():
    X, y, _ = _data()
    w = lr.fit_ols_normal_equations(X, y)
    np.testing.assert_allclose(w, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-10)


def test_fit_ols_pinv_handles_rank_deficiency():
    X, y, _ = _data(N=50, d=4)
    X = np.concatenate([X, X[:, :1] * 2.0], axis=1)  # (N, 5) last column is collinear
    w = lr.fit_ols_pinv(X, y)
    w_ref = np.linalg.lstsq(X, y, rcond=None)[0]  # also the minimum-norm solution
    np.testing.assert_allclose(w, w_ref, atol=1e-8)


def test_ols_gradient_finite_difference():
    X, y, _ = _data(N=30, d=4)
    w = np.random.default_rng(1).standard_normal(4)
    g = lr.ols_gradient(X, y, w)
    eps = 1e-6
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        fd = (lr.mse_loss(X, y, w + e) - lr.mse_loss(X, y, w - e)) / (2 * eps)
        np.testing.assert_allclose(g[j], fd, atol=1e-6)


def test_fit_ols_gradient_descent_converges_to_ols():
    X, y, _ = _data()
    w_gd = lr.fit_ols_gradient_descent(X, y, n_steps=3000)
    np.testing.assert_allclose(w_gd, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-6)


def test_fit_ridge_closed_form_shrinks_toward_zero():
    X, y, _ = _data()
    w0 = lr.fit_ols_normal_equations(X, y)
    prev = np.linalg.norm(w0)
    for lam in [0.1, 1.0, 10.0]:
        w = lr.fit_ridge(X, y, lam)
        # direct check of the normal equations with the λI term
        np.testing.assert_allclose((X.T @ X + lam * np.eye(5)) @ w, X.T @ y, atol=1e-8)
        assert np.linalg.norm(w) < prev
        prev = np.linalg.norm(w)


def test_fit_ridge_svd_equals_closed_form():
    X, y, _ = _data()
    for lam in [0.1, 1.0, 10.0]:
        np.testing.assert_allclose(lr.fit_ridge_svd(X, y, lam), lr.fit_ridge(X, y, lam), atol=1e-10)


def test_soft_threshold_known_values():
    np.testing.assert_allclose(lr.soft_threshold(np.array([3.0, -0.5, -2.0, 0.0]), 1.0), [2.0, 0.0, -1.0, 0.0])


def test_fit_lasso_coordinate_descent_is_sparse_and_optimal():
    rng = np.random.default_rng(0)
    N, d = 200, 10
    X = rng.standard_normal((N, d))
    w_true = np.zeros(d)
    w_true[:3] = [2.0, -1.5, 1.0]
    y = X @ w_true + 0.05 * rng.standard_normal(N)
    w = lr.fit_lasso_coordinate_descent(X, y, lam=0.1)
    assert np.all(np.abs(w[3:]) < 1e-6), w
    assert np.all(np.abs(w[:3]) > 0.5)
    obj = lr.lasso_objective(X, y, w, 0.1)
    assert obj <= lr.lasso_objective(X, y, lr.fit_ols_normal_equations(X, y), 0.1) + 1e-9
    assert obj <= lr.lasso_objective(X, y, np.zeros(d), 0.1) + 1e-9
    # KKT: for active coordinates, (1/N) x_j^T r = λ sign(w_j)
    r = y - X @ w
    for j in range(3):
        np.testing.assert_allclose(X[:, j] @ r / N, 0.1 * np.sign(w[j]), atol=1e-6)


def test_fit_lasso_with_zero_penalty_equals_ols():
    X, y, _ = _data(N=100, d=4)
    w = lr.fit_lasso_coordinate_descent(X, y, lam=0.0, n_sweeps=2000, tol=1e-12)
    np.testing.assert_allclose(w, lr.fit_ols_normal_equations(X, y), atol=1e-6)
