"""Probability primitives: samplers, densities, Gaussians, Bayes (pure NumPy).

Shapes: a batch of ``N`` points in ``d`` dimensions is ``(N, d)``; a covariance
is ``(d, d)``; probability vectors over ``K`` outcomes are ``(K,)``.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Special functions in pure NumPy (so the module needs no scipy)
# ---------------------------------------------------------------------------

_LANCZOS_G = 7
_LANCZOS_COEF = np.array(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
)


def log_gamma(x: np.ndarray | float) -> np.ndarray:
    """``log Gamma(x)`` for ``x > 0`` via the Lanczos approximation (rel. err ~1e-15).

    Needed for Beta / Dirichlet / Poisson / Binomial log-densities.
    Args:
        x: any shape, strictly positive.
    Returns:
        same shape as ``x``.
    """
    x = np.asarray(x, dtype=np.float64) - 1.0
    a = _LANCZOS_COEF[0] + np.zeros_like(x)  # same shape as x
    t = x + _LANCZOS_G + 0.5
    for i in range(1, _LANCZOS_G + 2):
        a = a + _LANCZOS_COEF[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


_erf = np.vectorize(math.erf, otypes=[np.float64])


def normal_cdf(z: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF ``Phi(z) = 1/2 (1 + erf(z / sqrt 2))``."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Samplers built from uniforms (the way a library does it)
# ---------------------------------------------------------------------------


def sample_categorical(p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling from a categorical distribution.

    Args:
        p: (K,) probabilities summing to one.
        n: number of samples.
    Returns:
        (n,) integer class indices in ``[0, K)``.
    """
    cdf = np.cumsum(p)  # (K,) monotone, last entry 1
    u = rng.random(n)  # (n,) uniforms in [0, 1)
    return np.searchsorted(cdf, u, side="right")  # (n,) first k with cdf[k] > u


def sample_gaussian_box_muller(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` standard normals from uniforms via Box-Muller.

    ``z0 = sqrt(-2 ln u1) cos(2 pi u2)``, ``z1 = sqrt(-2 ln u1) sin(2 pi u2)``.
    Returns:
        (n,).
    """
    m = (n + 1) // 2
    u1 = rng.random(m)  # (m,)
    u2 = rng.random(m)  # (m,)
    r = np.sqrt(-2.0 * np.log(1.0 - u1))  # (m,) 1-u1 avoids log(0)
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])  # (2m,)
    return z[:n]  # (n,)


def sample_multivariate_gaussian(mu: np.ndarray, Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Samples ``x = mu + L z`` with ``Sigma = L L^T`` (Cholesky) and ``z ~ N(0, I)``.

    Args:
        mu: (d,). Sigma: (d, d) PSD (positive definite for Cholesky).
    Returns:
        (n, d).
    """
    L = np.linalg.cholesky(Sigma)  # (d, d) lower triangular
    Z = rng.standard_normal((n, mu.shape[0]))  # (n, d)
    return mu[None, :] + Z @ L.T  # (n, d): each row is mu + L z


# ---------------------------------------------------------------------------
# Densities and log-densities
# ---------------------------------------------------------------------------


def gaussian_logpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Univariate ``log N(x; mu, sigma^2)`` elementwise over ``x`` of any shape."""
    z = (x - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * math.log(2 * math.pi)


def multivariate_gaussian_logpdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """``log N(x; mu, Sigma) = -1/2 [(x-mu)^T Sigma^{-1} (x-mu) + log det Sigma + d log 2 pi]``.

    Computed through the Cholesky factor for stability: ``Sigma = L L^T``,
    ``log det Sigma = 2 sum log diag(L)``, and ``Sigma^{-1} r`` via two triangular solves.

    Args:
        X: (N, d). mu: (d,). Sigma: (d, d) PD.
    Returns:
        (N,).
    """
    d = mu.shape[0]
    L = np.linalg.cholesky(Sigma)  # (d, d)
    R = X - mu[None, :]  # (N, d) residuals
    # Solve L a = r for each row -> a = L^{-1} r; then r^T Sigma^{-1} r = ||a||^2.
    A = np.linalg.solve(L, R.T)  # (d, N)
    maha = np.sum(A * A, axis=0)  # (N,) Mahalanobis distances squared
    logdet = 2.0 * np.sum(np.log(np.diag(L)))  # scalar
    return -0.5 * (maha + logdet + d * math.log(2 * math.pi))  # (N,)


def bernoulli_logpmf(x: np.ndarray, p: float) -> np.ndarray:
    """``log p^x (1-p)^{1-x}`` for ``x in {0, 1}`` (any shape)."""
    return x * np.log(p) + (1 - x) * np.log1p(-p)


def binomial_logpmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """``log C(n, k) + k log p + (n-k) log(1-p)``."""
    k = np.asarray(k, dtype=np.float64)
    log_choose = log_gamma(n + 1) - log_gamma(k + 1) - log_gamma(n - k + 1)
    return log_choose + k * np.log(p) + (n - k) * np.log1p(-p)


def poisson_logpmf(k: np.ndarray, lam: float) -> np.ndarray:
    """``log (lam^k e^{-lam} / k!) = k log lam - lam - log Gamma(k+1)``."""
    k = np.asarray(k, dtype=np.float64)
    return k * np.log(lam) - lam - log_gamma(k + 1)


def exponential_logpdf(x: np.ndarray, rate: float) -> np.ndarray:
    """``log(rate) - rate x`` for ``x >= 0``."""
    return np.log(rate) - rate * x


def beta_logpdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """``log Beta(x; a, b) = (a-1) log x + (b-1) log(1-x) - log B(a, b)``."""
    log_B = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
    return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - log_B


def dirichlet_logpdf(p: np.ndarray, alpha: np.ndarray) -> float:
    """``log Dir(p; alpha) = sum (alpha_k - 1) log p_k - log B(alpha)``.

    Args:
        p: (K,) on the simplex. alpha: (K,) positive concentrations.
    """
    log_B = np.sum(log_gamma(alpha)) - log_gamma(np.sum(alpha))
    return float(np.sum((alpha - 1) * np.log(p)) - log_B)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance ``Sigma = X_c^T X_c / (N - 1)``.

    Args:
        X: (N, d).
    Returns:
        (d, d) symmetric PSD.
    """
    Xc = X - X.mean(axis=0, keepdims=True)  # (N, d)
    return Xc.T @ Xc / (X.shape[0] - 1)  # (d, d)


def gaussian_condition(
    mu: np.ndarray, Sigma: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, x_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional of a joint Gaussian: ``p(x_a | x_b)``.

    ``mu_{a|b} = mu_a + Sigma_ab Sigma_bb^{-1} (x_b - mu_b)``
    ``Sigma_{a|b} = Sigma_aa - Sigma_ab Sigma_bb^{-1} Sigma_ba``  (the Schur complement)

    This is exactly the Kalman-filter update with ``x_a`` = state, ``x_b`` = measurement.

    Args:
        mu: (d,). Sigma: (d, d). idx_a: (da,) ints. idx_b: (db,) ints. x_b: (db,).
    Returns:
        mu_cond (da,), Sigma_cond (da, da).
    """
    S_aa = Sigma[np.ix_(idx_a, idx_a)]  # (da, da)
    S_ab = Sigma[np.ix_(idx_a, idx_b)]  # (da, db)
    S_bb = Sigma[np.ix_(idx_b, idx_b)]  # (db, db)
    gain = S_ab @ np.linalg.inv(S_bb)  # (da, db)  ("Kalman gain" shape)
    mu_cond = mu[idx_a] + gain @ (x_b - mu[idx_b])  # (da,)
    Sigma_cond = S_aa - gain @ S_ab.T  # (da, da)
    return mu_cond, Sigma_cond


def gaussian_marginal(mu: np.ndarray, Sigma: np.ndarray, idx_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal ``p(x_a)`` of a joint Gaussian: pick the sub-block. Returns (da,), (da, da)."""
    return mu[idx_a], Sigma[np.ix_(idx_a, idx_a)]


def bayes_posterior_discrete(prior: np.ndarray, likelihood: np.ndarray) -> tuple[np.ndarray, float]:
    """``p(z | x) = p(x | z) p(z) / sum_z' p(x | z') p(z')`` for a discrete latent.

    Args:
        prior: (K,) ``p(z)``. likelihood: (K,) ``p(x | z = k)`` for the observed x.
    Returns:
        posterior (K,), evidence ``p(x)`` (the normaliser).
    """
    joint = likelihood * prior  # (K,) p(x, z)
    evidence = float(joint.sum())  # p(x) -- the sum that becomes intractable for large z
    return joint / evidence, evidence


def thompson_sampling_bernoulli(true_p: np.ndarray, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
    """Beta-Bernoulli Thompson sampling on ``K`` arms.

    Posterior for arm k is ``Beta(1 + successes_k, 1 + failures_k)``; each round
    sample one draw per arm, play the argmax, update.

    Args:
        true_p: (K,) true success probabilities.
    Returns:
        (n_rounds,) arm played at each round.
    """
    K = true_p.shape[0]
    alpha = np.ones(K)  # (K,) successes + 1
    beta = np.ones(K)  # (K,) failures + 1
    played = np.zeros(n_rounds, dtype=np.int64)  # (n_rounds,)
    for t in range(n_rounds):
        theta = rng.beta(alpha, beta)  # (K,) one posterior sample per arm
        k = int(np.argmax(theta))
        reward = rng.random() < true_p[k]
        alpha[k] += reward
        beta[k] += 1 - reward
        played[t] = k
    return played
