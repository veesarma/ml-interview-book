"""One focused test per public symbol in mlbook.math.probability (select with -k <name>)."""

import numpy as np
from scipy import stats as sps
from scipy.special import gammaln

from mlbook.math import probability as pr


def test_log_gamma():
    x = np.array([0.5, 1.0, 2.5, 10.0, 100.5])
    assert np.allclose(pr.log_gamma(x), gammaln(x), rtol=1e-12)


def test_normal_cdf():
    z = np.linspace(-4, 4, 17)
    assert np.allclose(pr.normal_cdf(z), sps.norm.cdf(z), atol=1e-12)


def test_sample_categorical():
    rng = np.random.default_rng(0)
    p = np.array([0.1, 0.6, 0.3])
    s = pr.sample_categorical(p, 100_000, rng)
    freq = np.bincount(s, minlength=3) / s.shape[0]
    assert np.allclose(freq, p, atol=0.01)
    assert s.min() >= 0 and s.max() <= 2


def test_sample_gaussian_box_muller():
    rng = np.random.default_rng(0)
    z = pr.sample_gaussian_box_muller(200_000, rng)
    assert abs(z.mean()) < 0.01 and abs(z.var() - 1) < 0.01
    assert sps.kstest(z[:5000], "norm").pvalue > 0.01


def test_sample_multivariate_gaussian():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -2.0, 0.5])
    L = np.array([[1.0, 0, 0], [0.5, 1.0, 0], [-0.3, 0.2, 0.8]])
    Sigma = L @ L.T
    X = pr.sample_multivariate_gaussian(mu, Sigma, 100_000, rng)
    assert X.shape == (100_000, 3)
    assert np.allclose(X.mean(axis=0), mu, atol=0.02)
    assert np.allclose(np.cov(X, rowvar=False), Sigma, atol=0.03)


def test_multivariate_gaussian_logpdf():
    mu = np.array([1.0, -2.0, 0.5])
    L = np.array([[1.0, 0, 0], [0.5, 1.0, 0], [-0.3, 0.2, 0.8]])
    Sigma = L @ L.T
    X = np.random.randn(10, 3)
    assert np.allclose(pr.multivariate_gaussian_logpdf(X, mu, Sigma), sps.multivariate_normal(mu, Sigma).logpdf(X))


def test_gaussian_logpdf():
    assert np.isclose(pr.gaussian_logpdf(0.3, 1.0, 2.0), sps.norm(1.0, 2.0).logpdf(0.3))


def test_bernoulli_logpmf():
    x = np.array([0, 1, 1, 0])
    assert np.allclose(pr.bernoulli_logpmf(x, 0.3), sps.bernoulli(0.3).logpmf(x))


def test_binomial_logpmf():
    k = np.arange(0, 11)
    assert np.allclose(pr.binomial_logpmf(k, 10, 0.3), sps.binom(10, 0.3).logpmf(k))


def test_poisson_logpmf():
    k = np.arange(0, 20)
    assert np.allclose(pr.poisson_logpmf(k, 2.5), sps.poisson(2.5).logpmf(k))
    p = np.exp(pr.poisson_logpmf(np.arange(0, 200), 2.5))
    assert np.isclose((p * np.arange(200)).sum(), 2.5)  # mean = lambda


def test_exponential_logpdf():
    x = np.linspace(0.05, 3, 7)
    assert np.allclose(pr.exponential_logpdf(x, 1.7), sps.expon(scale=1 / 1.7).logpdf(x))


def test_beta_logpdf():
    x = np.linspace(0.05, 0.95, 7)
    assert np.allclose(pr.beta_logpdf(x, 2.0, 5.0), sps.beta(2.0, 5.0).logpdf(x))


def test_dirichlet_logpdf():
    p = np.array([0.2, 0.3, 0.5])
    alpha = np.array([1.5, 2.0, 3.0])
    assert np.isclose(pr.dirichlet_logpdf(p, alpha), sps.dirichlet(alpha).logpdf(p))


def test_covariance_matrix():
    X = np.random.randn(500, 4) @ np.random.randn(4, 4)
    assert np.allclose(pr.covariance_matrix(X), np.cov(X, rowvar=False))


def test_gaussian_condition():
    rng = np.random.default_rng(1)
    mu = np.array([0.0, 1.0])
    Sigma = np.array([[2.0, 1.2], [1.2, 1.5]])
    mu_c, Sigma_c = pr.gaussian_condition(mu, Sigma, np.array([0]), np.array([1]), np.array([2.0]))
    assert np.isclose(mu_c[0], 0.0 + 1.2 / 1.5 * (2.0 - 1.0))  # mu_a + S_ab S_bb^-1 (x_b - mu_b)
    assert np.isclose(Sigma_c[0, 0], 2.0 - 1.2**2 / 1.5)  # Schur complement
    X = pr.sample_multivariate_gaussian(mu, Sigma, 400_000, rng)
    sel = np.abs(X[:, 1] - 2.0) < 0.02
    assert abs(X[sel, 0].mean() - mu_c[0]) < 0.05


def test_gaussian_marginal():
    mu = np.array([0.0, 1.0, 2.0])
    Sigma = np.array([[2.0, 1.2, 0.1], [1.2, 1.5, 0.3], [0.1, 0.3, 1.0]])
    m, S = pr.gaussian_marginal(mu, Sigma, np.array([1, 2]))
    assert np.allclose(m, [1.0, 2.0]) and np.allclose(S, [[1.5, 0.3], [0.3, 1.0]])


def test_bayes_posterior_discrete():
    prior = np.array([0.01, 0.99])  # disease, healthy
    likelihood = np.array([0.95, 0.05])  # P(positive | z)
    post, evidence = pr.bayes_posterior_discrete(prior, likelihood)
    assert np.isclose(evidence, 0.01 * 0.95 + 0.99 * 0.05)
    assert np.isclose(post[0], 0.0095 / evidence)
    assert np.isclose(post.sum(), 1.0)


def test_thompson_sampling_bernoulli():
    rng = np.random.default_rng(0)
    played = pr.thompson_sampling_bernoulli(np.array([0.2, 0.5, 0.8]), 2000, rng)
    assert played.shape == (2000,)
    assert np.mean(played[-500:] == 2) > 0.9  # converges on the best arm
