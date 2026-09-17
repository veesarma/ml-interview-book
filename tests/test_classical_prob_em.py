"""Tests for naive_bayes.py, gda.py, gmm.py — one focused test per retype target."""
import numpy as np

from mlbook.classical.gda import GDA, lda_as_logistic
from mlbook.classical.gmm import GMM, log_gaussian, logsumexp
from mlbook.classical.logistic_regression import sigmoid
from mlbook.classical.naive_bayes import GaussianNB, MultinomialNB


def _two_blobs(seed=0, N=400):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    X = np.array([[0, 0], [3, 3.0]])[y] + rng.standard_normal((N, 2))
    return X, y


def test_multinomial_nb_laplace_smoothing_known_values():
    X = np.array([[2, 0], [1, 1], [0, 3]])
    y = np.array([0, 0, 1])
    nb = MultinomialNB(alpha=1.0).fit(X, y)
    np.testing.assert_allclose(np.exp(nb.log_theta[0]), [4 / 6, 2 / 6])
    np.testing.assert_allclose(np.exp(nb.log_theta[1]), [1 / 5, 4 / 5])
    np.testing.assert_allclose(np.exp(nb.log_prior), [2 / 3, 1 / 3])
    assert nb.predict(np.array([[3, 0]]))[0] == 0
    assert nb.predict(np.array([[0, 3]]))[0] == 1


def test_gaussian_nb_separates_blobs():
    X, y = _two_blobs()
    nb = GaussianNB().fit(X, y)
    assert (nb.predict(X) == y).mean() > 0.95
    np.testing.assert_allclose(nb.mu[1], X[y == 1].mean(axis=0))


def test_gda_lda_and_qda_separate_blobs():
    X, y = _two_blobs()
    assert (GDA(shared_cov=True).fit(X, y).predict(X) == y).mean() > 0.95
    assert (GDA(shared_cov=False).fit(X, y).predict(X) == y).mean() > 0.95
    lda = GDA(shared_cov=True).fit(X, y)
    np.testing.assert_allclose(lda.Sigma[0], lda.Sigma[1])


def test_lda_as_logistic_matches_posterior():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    X = np.array([[0, 0], [2, 1.0]])[y] + rng.standard_normal((500, 2)) @ np.array([[1.0, 0.3], [0.0, 0.8]])
    lda = GDA(shared_cov=True).fit(X, y)
    lp = lda.log_posterior(X)  # (N, 2)
    post1 = np.exp(lp[:, 1] - np.logaddexp(lp[:, 0], lp[:, 1]))
    w, b = lda_as_logistic(lda)
    np.testing.assert_allclose(post1, sigmoid(X @ w + b), atol=1e-10)


def test_log_gaussian_and_logsumexp():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    mu = rng.standard_normal(3)
    A = rng.standard_normal((3, 3))
    S = A @ A.T + np.eye(3)
    ref = -0.5 * (3 * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + np.einsum("ni,ij,nj->n", X - mu, np.linalg.inv(S), X - mu))
    np.testing.assert_allclose(log_gaussian(X, mu, S), ref, atol=1e-10)
    A2 = np.array([[1000.0, 1000.0], [0.0, np.log(3.0)]])
    np.testing.assert_allclose(logsumexp(A2, axis=1), [1000.0 + np.log(2.0), np.log(4.0)])


def _mixture(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    pi_true = np.array([0.3, 0.7])
    mu_true = np.array([[-3.0, 0.0], [3.0, 1.0]])
    S_true = np.array([[[1.0, 0.3], [0.3, 0.5]], [[0.6, -0.2], [-0.2, 1.2]]])
    z = rng.random(n) < pi_true[1]
    X = np.where(z[:, None], rng.multivariate_normal(mu_true[1], S_true[1], n), rng.multivariate_normal(mu_true[0], S_true[0], n))
    return X, pi_true, mu_true, S_true


def test_gmm_e_step_and_m_step_closed_forms():
    X, _, mu_true, S_true = _mixture()
    gmm = GMM(k=2)
    gmm.pi, gmm.mu, gmm.Sigma = np.array([0.5, 0.5]), mu_true.copy(), S_true.copy()
    R, ll = gmm.e_step(X)
    np.testing.assert_allclose(R.sum(axis=1), 1.0)
    assert np.isfinite(ll)
    gmm.m_step(X, R)
    Nk = R.sum(axis=0)
    np.testing.assert_allclose(gmm.pi, Nk / len(X))
    np.testing.assert_allclose(gmm.mu, (R.T @ X) / Nk[:, None])


def test_gmm_fit_loglik_monotone_and_recovers_parameters():
    X, pi_true, mu_true, S_true = _mixture()
    gmm = GMM(k=2, n_iters=200, seed=0).fit(X)
    hist = np.array(gmm.history_)
    assert np.all(np.diff(hist) >= -1e-6), hist
    order = np.argsort(gmm.mu[:, 0])
    np.testing.assert_allclose(gmm.pi[order], pi_true, atol=0.05)
    np.testing.assert_allclose(gmm.mu[order], mu_true, atol=0.15)
    np.testing.assert_allclose(gmm.Sigma[order], S_true, atol=0.2)
