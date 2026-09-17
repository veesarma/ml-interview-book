import numpy as np
from scipy import stats as sps

from mlbook.math import stats as st


def test_mle_gaussian_is_biased_variance():
    x = np.random.randn(1000) * 2 + 3
    mu, var = st.mle_gaussian(x)
    assert np.isclose(mu, x.mean())
    assert np.isclose(var, x.var(ddof=0))
    assert var < x.var(ddof=1)


def test_map_mean_shrinks_toward_prior():
    x = np.array([10.0, 12.0, 11.0])
    mu_map = st.map_gaussian_mean(x, sigma=1.0, mu0=0.0, tau=1.0)
    # precision-weighted: (33/1 + 0) / (3 + 1) = 8.25
    assert np.isclose(mu_map, 8.25)
    assert st.map_gaussian_mean(x, sigma=1.0, mu0=0.0, tau=1e6) == np.mean(x) or np.isclose(
        st.map_gaussian_mean(x, sigma=1.0, mu0=0.0, tau=1e6), x.mean(), atol=1e-4
    )


def test_ridge_closed_form_matches_gradient_zero():
    X, y, lam = np.random.randn(30, 4), np.random.randn(30), 0.7
    w = st.ridge_closed_form(X, y, lam)
    grad = 2 * X.T @ (X @ w - y) + 2 * lam * w
    assert np.allclose(grad, 0.0, atol=1e-10)


def test_bias_variance_decomposition_sums():
    preds = np.random.randn(50, 20) + 0.3
    y = np.zeros(20)
    out = st.bias_variance_decomposition(preds, y, noise_var=0.1)
    assert np.isclose(out["bias_sq"] + out["variance"] + out["noise"], out["total"])
    # E[(f_hat - y)^2] over models and points equals bias^2 + variance (no noise term in this MC)
    mse = np.mean((preds - y[None, :]) ** 2)
    assert np.isclose(mse, out["bias_sq"] + out["variance"])


def test_z_critical_and_ci():
    assert np.isclose(st.z_critical(0.95), 1.959964, atol=1e-5)
    x = np.random.randn(500)
    lo, hi = st.confidence_interval_mean(x, 0.95)
    lo_ref, hi_ref = sps.norm.interval(0.95, loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(500))
    assert np.isclose(lo, lo_ref, atol=1e-5) and np.isclose(hi, hi_ref, atol=1e-5)


def test_two_proportion_z_matches_statsmodels_formula():
    out = st.two_proportion_z_test(k_a=500, n_a=10_000, k_b=560, n_b=10_000)
    p_pool = 1060 / 20_000
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / 10_000)
    z = (0.056 - 0.05) / se
    assert np.isclose(out["z"], z)
    assert np.isclose(out["p_value"], 2 * sps.norm.sf(abs(z)))


def test_welch_t_matches_scipy_for_large_n():
    a = np.random.randn(5000)
    b = np.random.randn(6000) * 1.5 + 0.05
    out = st.welch_t_statistic(a, b)
    ref = sps.ttest_ind(b, a, equal_var=False)
    assert np.isclose(out["t"], ref.statistic)
    assert np.isclose(out["p_value"], ref.pvalue, atol=1e-3)


def test_sample_size_reasonable():
    n = st.sample_size_two_proportions(p0=0.05, mde_abs=0.005, alpha=0.05, power=0.8)
    assert 28_000 < n < 33_000  # textbook answer ~ 31k per arm


def test_bootstrap_ci_covers_mean():
    x = np.random.randn(300) + 1.0
    lo, hi = st.bootstrap_ci(x, np.mean, n_boot=500)
    assert lo < 1.0 < hi


def test_beta_binomial_posterior_and_ece():
    assert st.beta_binomial_posterior(1, 1, 7, 3) == (8, 4)
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert np.isclose(st.expected_calibration_error(probs, labels, n_bins=10), 0.1)


def test_clt_sample_means_are_gaussian():
    rng = np.random.default_rng(0)
    means = st.clt_sample_means(lambda size: rng.exponential(1.0, size), n_per_sample=100, n_samples=20_000)
    z = (means - 1.0) / (1.0 / np.sqrt(100))
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
    assert abs(sps.skew(z)) < 0.3  # exponential skew (2) shrinks like 1/sqrt(n)
