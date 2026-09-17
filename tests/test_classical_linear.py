import numpy as np

from mlbook.classical import linear_regression as lr


def _data(N=200, d=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))  # (N, d)
    w_true = rng.standard_normal(d)  # (d,)
    y = X @ w_true + noise * rng.standard_normal(N)  # (N,)
    return X, y, w_true


def test_normal_equations_match_lstsq():
    X, y, _ = _data()
    w = lr.fit_ols_normal_equations(X, y)
    w_ref = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(w, w_ref, atol=1e-10)


def test_pinv_handles_rank_deficient_and_matches_lstsq():
    X, y, _ = _data(N=50, d=4)
    X = np.concatenate([X, X[:, :1] * 2.0], axis=1)  # (N, 5) last column is collinear
    w = lr.fit_ols_pinv(X, y)
    w_ref = np.linalg.lstsq(X, y, rcond=None)[0]  # also the minimum-norm solution
    np.testing.assert_allclose(w, w_ref, atol=1e-8)
    np.testing.assert_allclose(X @ w, X @ w_ref, atol=1e-8)


def test_ols_gradient_finite_difference():
    X, y, _ = _data(N=30, d=4)
    w = np.random.default_rng(1).standard_normal(4)
    g = lr.ols_gradient(X, y, w)
    eps = 1e-6
    g_fd = np.zeros_like(w)
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        g_fd[j] = (lr.mse_loss(X, y, w + e) - lr.mse_loss(X, y, w - e)) / (2 * eps)
    np.testing.assert_allclose(g, g_fd, atol=1e-6)


def test_gradient_descent_converges_to_ols():
    X, y, _ = _data()
    w_gd = lr.fit_ols_gradient_descent(X, y, n_steps=3000)
    w_ref = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(w_gd, w_ref, atol=1e-6)


def test_ridge_closed_form_equals_svd_form_and_shrinks():
    X, y, _ = _data()
    w0 = lr.fit_ols_normal_equations(X, y)
    for lam in [0.1, 1.0, 10.0]:
        w_a = lr.fit_ridge(X, y, lam)
        w_b = lr.fit_ridge_svd(X, y, lam)
        np.testing.assert_allclose(w_a, w_b, atol=1e-10)
        assert np.linalg.norm(w_a) < np.linalg.norm(w0)


def test_lasso_soft_threshold_and_sparsity():
    np.testing.assert_allclose(lr.soft_threshold(np.array([3.0, -0.5, -2.0]), 1.0), [2.0, 0.0, -1.0])
    rng = np.random.default_rng(0)
    N, d = 200, 10
    X = rng.standard_normal((N, d))
    w_true = np.zeros(d)
    w_true[:3] = [2.0, -1.5, 1.0]
    y = X @ w_true + 0.05 * rng.standard_normal(N)
    w = lr.fit_lasso_coordinate_descent(X, y, lam=0.1)
    assert np.all(np.abs(w[3:]) < 1e-6), w
    assert np.all(np.abs(w[:3]) > 0.5)
    # objective is no worse than at the OLS solution and at zero
    obj = lr.lasso_objective(X, y, w, 0.1)
    assert obj <= lr.lasso_objective(X, y, lr.fit_ols_normal_equations(X, y), 0.1) + 1e-9
    assert obj <= lr.lasso_objective(X, y, np.zeros(d), 0.1) + 1e-9


def test_lasso_with_zero_penalty_equals_ols():
    X, y, _ = _data(N=100, d=4)
    w = lr.fit_lasso_coordinate_descent(X, y, lam=0.0, n_sweeps=2000, tol=1e-12)
    np.testing.assert_allclose(w, lr.fit_ols_normal_equations(X, y), atol=1e-6)
