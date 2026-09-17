"""Tests for kernels.py and svm.py — one focused test per retype target."""
import numpy as np

from mlbook.classical import kernels as kn
from mlbook.classical.svm import SVM, dual_objective, fit_linear_svm_primal, hinge_loss


def test_kernels_psd_and_polynomial_matches_explicit_features():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    for K in [kn.linear_kernel(X, X), kn.polynomial_kernel(X, X, 2), kn.rbf_kernel(X, X, 0.5)]:
        assert np.linalg.eigvalsh(K).min() > -1e-8
    np.testing.assert_allclose(np.diag(kn.rbf_kernel(X, X, 0.5)), 1.0)

    def phi(x):  # explicit feature map of (x·y + 1)^2 in 3-D
        return np.concatenate([[1.0], np.sqrt(2) * x, x**2, [np.sqrt(2) * x[0] * x[1], np.sqrt(2) * x[0] * x[2], np.sqrt(2) * x[1] * x[2]]])

    Phi = np.stack([phi(x) for x in X])  # (20, 10)
    np.testing.assert_allclose(kn.polynomial_kernel(X, X, degree=2), Phi @ Phi.T, atol=1e-10)


def test_kernel_ridge_linear_equals_primal_ridge_and_rbf_fits_sine():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1 * rng.standard_normal(50)
    lam = 0.5
    krr = kn.KernelRidge(kn.linear_kernel, lam).fit(X, y)
    w_primal = np.linalg.solve(X.T @ X + lam * np.eye(4), X.T @ y)
    np.testing.assert_allclose(krr.predict(X), X @ w_primal, atol=1e-8)
    Xs = np.linspace(-3, 3, 100)[:, None]
    ys = np.sin(Xs[:, 0])
    krr_rbf = kn.KernelRidge(lambda A, B: kn.rbf_kernel(A, B, 1.0), 1e-3).fit(Xs, ys)
    assert np.mean((krr_rbf.predict(Xs) - ys) ** 2) < 1e-3


def test_nadaraya_watson_is_attention():
    rng = np.random.default_rng(0)
    Q, X, V = rng.standard_normal((3, 4)), rng.standard_normal((7, 4)), rng.standard_normal((7, 2))
    out = kn.nadaraya_watson(Q, X, V, lambda A, B: np.exp(A @ B.T / 2.0))
    scores = Q @ X.T / 2.0  # (3, 7)
    attn = np.exp(scores - scores.max(1, keepdims=True))
    attn /= attn.sum(1, keepdims=True)
    np.testing.assert_allclose(out, attn @ V, atol=1e-10)


def _separable(seed=0, n=60):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    X = X + 0.4 * y[:, None]
    return X, y


def test_svm_dual_satisfies_kkt():
    X, y = _separable()
    svm = SVM(kn.linear_kernel, C=10.0, max_passes=30).fit(X, y)
    assert (svm.predict(X) == y).all()
    assert abs((svm.alpha * y).sum()) < 1e-8  # Σ α_i y_i = 0
    assert np.all((svm.alpha >= -1e-9) & (svm.alpha <= 10.0 + 1e-9))
    margins = y * svm.decision_function(X)
    assert np.all(margins[svm.support_] <= 1.0 + 1e-2)  # support vectors on/inside the margin
    free = (svm.alpha > 1e-6) & (svm.alpha < 10.0 - 1e-6)
    np.testing.assert_allclose(margins[free], 1.0, atol=5e-2)  # free SVs sit exactly on the margin


def test_dual_objective_and_hinge_loss_values():
    y = np.array([1.0, -1.0])
    K = np.eye(2)
    np.testing.assert_allclose(dual_objective(np.array([1.0, 1.0]), y, K), 2.0 - 0.5 * 2.0)
    X = np.array([[1.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_allclose(hinge_loss(X, y, np.array([0.5, 0.0]), 0.0, C=1.0), 0.125 + 2 * 0.5)


def test_fit_linear_svm_primal_separates_and_lowers_hinge():
    X, y = _separable()
    w, b = fit_linear_svm_primal(X, y, C=10.0, lr=1e-3, n_steps=3000)
    assert (np.sign(X @ w + b) == y).all()
    assert hinge_loss(X, y, w, b, 10.0) < hinge_loss(X, y, np.zeros(2), 0.0, 10.0)


def test_rbf_svm_solves_xor():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (150, 2))
    y = np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 1, -1)
    svm = SVM(lambda A, B: kn.rbf_kernel(A, B, gamma=3.0), C=5.0, max_passes=20).fit(X, y)
    assert (svm.predict(X) == y).mean() > 0.95
