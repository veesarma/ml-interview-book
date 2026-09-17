# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/math/probability.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k probability -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py math/probability --force

"""Probability primitives: samplers, densities, Gaussians, Bayes (pure NumPy).

Shapes: a batch of ``N`` points in ``d`` dimensions is ``(N, d)``; a covariance
is ``(d, d)``; probability vectors over ``K`` outcomes are ``(K,)``.
"""
from __future__ import annotations
import math
import numpy as np
_LANCZOS_G = 7
_LANCZOS_COEF = np.array([0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019572e-06, 1.5056327351493116e-07])

def log_gamma(x: np.ndarray | float) -> np.ndarray:
    """``log Gamma(x)`` for ``x > 0`` via the Lanczos approximation (rel. err ~1e-15).

    Needed for Beta / Dirichlet / Poisson / Binomial log-densities.
    Args:
        x: any shape, strictly positive.
    Returns:
        same shape as ``x``.
    """
    raise NotImplementedError('TODO: implement log_gamma (see the reference in src/mlbook)')
_erf = np.vectorize(math.erf, otypes=[np.float64])

def normal_cdf(z: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF ``Phi(z) = 1/2 (1 + erf(z / sqrt 2))``."""
    raise NotImplementedError('TODO: implement normal_cdf (see the reference in src/mlbook)')

def sample_categorical(p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling from a categorical distribution.

    Args:
        p: (K,) probabilities summing to one.
        n: number of samples.
    Returns:
        (n,) integer class indices in ``[0, K)``.
    """
    raise NotImplementedError('TODO: implement sample_categorical (see the reference in src/mlbook)')

def sample_gaussian_box_muller(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` standard normals from uniforms via Box-Muller.

    ``z0 = sqrt(-2 ln u1) cos(2 pi u2)``, ``z1 = sqrt(-2 ln u1) sin(2 pi u2)``.
    Returns:
        (n,).
    """
    raise NotImplementedError('TODO: implement sample_gaussian_box_muller (see the reference in src/mlbook)')

def sample_multivariate_gaussian(mu: np.ndarray, Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Samples ``x = mu + L z`` with ``Sigma = L L^T`` (Cholesky) and ``z ~ N(0, I)``.

    Args:
        mu: (d,). Sigma: (d, d) PSD (positive definite for Cholesky).
    Returns:
        (n, d).
    """
    raise NotImplementedError('TODO: implement sample_multivariate_gaussian (see the reference in src/mlbook)')

def gaussian_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Univariate ``log N(x; mu, sigma^2)`` elementwise over ``x`` of any shape."""
    raise NotImplementedError('TODO: implement gaussian_logpdf (see the reference in src/mlbook)')

def multivariate_gaussian_logpdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """``log N(x; mu, Sigma) = -1/2 [(x-mu)^T Sigma^{-1} (x-mu) + log det Sigma + d log 2 pi]``.

    Computed through the Cholesky factor for stability: ``Sigma = L L^T``,
    ``log det Sigma = 2 sum log diag(L)``, and ``Sigma^{-1} r`` via two triangular solves.

    Args:
        X: (N, d). mu: (d,). Sigma: (d, d) PD.
    Returns:
        (N,).
    """
    raise NotImplementedError('TODO: implement multivariate_gaussian_logpdf (see the reference in src/mlbook)')

def bernoulli_logpmf(x: np.ndarray, p: float) -> np.ndarray:
    """``log p^x (1-p)^{1-x}`` for ``x in {0, 1}`` (any shape)."""
    raise NotImplementedError('TODO: implement bernoulli_logpmf (see the reference in src/mlbook)')

def binomial_logpmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """``log C(n, k) + k log p + (n-k) log(1-p)``."""
    raise NotImplementedError('TODO: implement binomial_logpmf (see the reference in src/mlbook)')

def poisson_logpmf(k: np.ndarray, lam: float) -> np.ndarray:
    """``log (lam^k e^{-lam} / k!) = k log lam - lam - log Gamma(k+1)``."""
    raise NotImplementedError('TODO: implement poisson_logpmf (see the reference in src/mlbook)')

def exponential_logpdf(x: np.ndarray, rate: float) -> np.ndarray:
    """``log(rate) - rate x`` for ``x >= 0``."""
    raise NotImplementedError('TODO: implement exponential_logpdf (see the reference in src/mlbook)')

def beta_logpdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """``log Beta(x; a, b) = (a-1) log x + (b-1) log(1-x) - log B(a, b)``."""
    raise NotImplementedError('TODO: implement beta_logpdf (see the reference in src/mlbook)')

def dirichlet_logpdf(p: np.ndarray, alpha: np.ndarray) -> float:
    """``log Dir(p; alpha) = sum (alpha_k - 1) log p_k - log B(alpha)``.

    Args:
        p: (K,) on the simplex. alpha: (K,) positive concentrations.
    """
    raise NotImplementedError('TODO: implement dirichlet_logpdf (see the reference in src/mlbook)')

def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance ``Sigma = X_c^T X_c / (N - 1)``.

    Args:
        X: (N, d).
    Returns:
        (d, d) symmetric PSD.
    """
    raise NotImplementedError('TODO: implement covariance_matrix (see the reference in src/mlbook)')

def gaussian_condition(mu: np.ndarray, Sigma: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, x_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conditional of a joint Gaussian: ``p(x_a | x_b)``.

    ``mu_{a|b} = mu_a + Sigma_ab Sigma_bb^{-1} (x_b - mu_b)``
    ``Sigma_{a|b} = Sigma_aa - Sigma_ab Sigma_bb^{-1} Sigma_ba``  (the Schur complement)

    This is exactly the Kalman-filter update with ``x_a`` = state, ``x_b`` = measurement.

    Args:
        mu: (d,). Sigma: (d, d). idx_a: (da,) ints. idx_b: (db,) ints. x_b: (db,).
    Returns:
        mu_cond (da,), Sigma_cond (da, da).
    """
    raise NotImplementedError('TODO: implement gaussian_condition (see the reference in src/mlbook)')

def gaussian_marginal(mu: np.ndarray, Sigma: np.ndarray, idx_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal ``p(x_a)`` of a joint Gaussian: pick the sub-block. Returns (da,), (da, da)."""
    raise NotImplementedError('TODO: implement gaussian_marginal (see the reference in src/mlbook)')

def bayes_posterior_discrete(prior: np.ndarray, likelihood: np.ndarray) -> tuple[np.ndarray, float]:
    """``p(z | x) = p(x | z) p(z) / sum_z' p(x | z') p(z')`` for a discrete latent.

    Args:
        prior: (K,) ``p(z)``. likelihood: (K,) ``p(x | z = k)`` for the observed x.
    Returns:
        posterior (K,), evidence ``p(x)`` (the normaliser).
    """
    raise NotImplementedError('TODO: implement bayes_posterior_discrete (see the reference in src/mlbook)')

def thompson_sampling_bernoulli(true_p: np.ndarray, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
    """Beta-Bernoulli Thompson sampling on ``K`` arms.

    Posterior for arm k is ``Beta(1 + successes_k, 1 + failures_k)``; each round
    sample one draw per arm, play the argmax, update.

    Args:
        true_p: (K,) true success probabilities.
    Returns:
        (n_rounds,) arm played at each round.
    """
    raise NotImplementedError('TODO: implement thompson_sampling_bernoulli (see the reference in src/mlbook)')
