import numpy as np

from mlbook.classical.gda import GDA, lda_as_logistic
from mlbook.classical.gmm import GMM, log_gaussian
from mlbook.classical.logistic_regression import sigmoid
from mlbook.classical.naive_bayes import GaussianNB, MultinomialNB


def test_multinomial_nb_laplace_smoothing_known_values():
    X = np.array([[2, 0], [1, 1], [0, 3]])
    y = np.array([0, 0, 1])
    nb = MultinomialNB(alpha=1.0).fit(X, y)
    # class 0 counts: [3, 1], total 4, d=2 -> θ = [(3+1)/(4+2), (1+1)/(4+2)]
    np.testing.assert_allclose(np.exp(nb.log_theta[0]), [4 / 6, 2 / 6])
    np.testing.assert_allclose(np.exp(nb.log_theta[1]), [1 / 5, 4 / 5])
    assert nb.predict(np.array([[3, 0]]))[0] == 0
    assert nb.predict(np.array([[0, 3]]))[0] == 1


def test_gaussian_nb_and_gda_separate_blobs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    X = np.array([[0, 0], [3, 3.0]])[y] + rng.standard_normal((400, 2))
    assert (GaussianNB().fit(X, y).predict(X) == y).mean() > 0.95
    assert (GDA(shared_cov=True).fit(X, y).predict(X) == y).mean() > 0.95
    assert (GDA(shared_cov=False).fit(X, y).predict(X) == y).mean() > 0.95


def test_lda_posterior_is_logistic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    X = np.array([[0, 0], [2, 1.0]])[y] + rng.standard_normal((500, 2)) @ np.array([[1.0, 0.3], [0.0, 0.8]])
    lda = GDA(shared_cov=True).fit(X, y)
    lp = lda.log_posterior(X)  # (N, 2)
    post1 = np.exp(lp[:, 1] - np.logaddexp(lp[:, 0], lp[:, 1]))
    w, b = lda_as_logistic(lda)
    np.testing.assert_allclose(post1, sigmoid(X @ w + b), atol=1e-10)


def test_log_gaussian_matches_formula():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    mu = rng.standard_normal(3)
    A = rng.standard_normal((3, 3))
    S = A @ A.T + np.eye(3)
    ref = -0.5 * (3 * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + np.einsum("ni,ij,nj->n", X - mu, np.linalg.inv(S), X - mu))
    np.testing.assert_allclose(log_gaussian(X, mu, S), ref, atol=1e-10)


def test_gmm_em_loglik_monotone_and_recovers_parameters():
    rng = np.random.default_rng(0)
    pi_true = np.array([0.3, 0.7])
    mu_true = np.array([[-3.0, 0.0], [3.0, 1.0]])
    S_true = np.array([[[1.0, 0.3], [0.3, 0.5]], [[0.6, -0.2], [-0.2, 1.2]]])
    z = rng.random(1500) < pi_true[1]
    X = np.where(
        z[:, None],
        rng.multivariate_normal(mu_true[1], S_true[1], 1500),
        rng.multivariate_normal(mu_true[0], S_true[0], 1500),
    )
    gmm = GMM(k=2, n_iters=200, seed=0).fit(X)
    hist = np.array(gmm.history_)
    assert np.all(np.diff(hist) >= -1e-6), hist
    order = np.argsort(gmm.mu[:, 0])
    np.testing.assert_allclose(gmm.pi[order], pi_true, atol=0.05)
    np.testing.assert_allclose(gmm.mu[order], mu_true, atol=0.15)
    np.testing.assert_allclose(gmm.Sigma[order], S_true, atol=0.2)
    R = gmm.predict_proba(X)
    np.testing.assert_allclose(R.sum(axis=1), 1.0)
