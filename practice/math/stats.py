# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/math/stats.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k stats -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py math/stats --force

"""Statistics: MLE/MAP, bias-variance, intervals, A/B tests, bootstrap, calibration.

Pure NumPy; the normal CDF comes from ``mlbook.math.probability``.
"""
from __future__ import annotations
import math
import numpy as np
from mlbook.math.probability import normal_cdf

def mle_gaussian(x: np.ndarray) -> tuple[float, float]:
    """MLE of a univariate Gaussian: ``mu = mean(x)``, ``sigma^2 = mean((x-mu)^2)`` (biased, 1/N).

    Args:
        x: (N,).
    """
    raise NotImplementedError('TODO: implement mle_gaussian (see the reference in src/mlbook)')

def map_gaussian_mean(x: np.ndarray, sigma: float, mu0: float, tau: float) -> float:
    """MAP of the mean with likelihood ``N(x; mu, sigma^2)`` and prior ``N(mu; mu0, tau^2)``.

    ``mu_MAP = (sum x / sigma^2 + mu0 / tau^2) / (N / sigma^2 + 1 / tau^2)``:
    a precision-weighted average of the data mean and the prior mean.

    Args:
        x: (N,).
    """
    raise NotImplementedError('TODO: implement map_gaussian_mean (see the reference in src/mlbook)')

def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge = MAP with Gaussian prior: ``w = (X^T X + lam I)^{-1} X^T y``.

    Args:
        X: (N, d). y: (N,). lam: L2 strength = sigma^2 / tau^2 in the Bayesian reading.
    Returns:
        (d,).
    """
    raise NotImplementedError('TODO: implement ridge_closed_form (see the reference in src/mlbook)')

def bias_variance_decomposition(predictions: np.ndarray, y_true: np.ndarray, noise_var: float=0.0) -> dict:
    """Decompose expected squared error over ``M`` re-trained models.

    ``E[(f_hat - y)^2] = (E f_hat - f)^2 + Var[f_hat] + sigma^2``.

    Args:
        predictions: (M, N) predictions of M independently trained models on N test points.
        y_true: (N,) noiseless targets ``f(x)``.
    Returns:
        dict with mean ``bias_sq``, ``variance``, ``noise``, ``total`` over the N points.
    """
    raise NotImplementedError('TODO: implement bias_variance_decomposition (see the reference in src/mlbook)')

def z_critical(confidence: float) -> float:
    """Two-sided standard-normal critical value, e.g. 1.96 for 0.95 (bisection on Phi)."""
    raise NotImplementedError('TODO: implement z_critical (see the reference in src/mlbook)')

def confidence_interval_mean(x: np.ndarray, confidence: float=0.95) -> tuple[float, float]:
    """Normal-approximation CI for a mean: ``mean +- z * s / sqrt(N)`` (CLT).

    Args:
        x: (N,), N large enough for the CLT (rule of thumb: > 30, non-heavy-tailed).
    """
    raise NotImplementedError('TODO: implement confidence_interval_mean (see the reference in src/mlbook)')

def two_proportion_z_test(k_a: int, n_a: int, k_b: int, n_b: int) -> dict:
    """A/B test on conversion rates: pooled two-proportion z-test.

    ``z = (p_b - p_a) / sqrt(p (1-p) (1/n_a + 1/n_b))``, ``p`` pooled.
    Two-sided p-value ``2 (1 - Phi(|z|))``.

    Args:
        k_a, n_a: conversions and users in control; k_b, n_b: in treatment.
    Returns:
        dict with lift ``p_b - p_a``, ``z``, ``p_value``, ``se``.
    """
    raise NotImplementedError('TODO: implement two_proportion_z_test (see the reference in src/mlbook)')

def welch_t_statistic(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    """Welch's t for unequal variances: ``t = (mean_b - mean_a) / sqrt(s_a^2/n_a + s_b^2/n_b)``.

    The p-value uses the normal approximation to the t-distribution, which is
    accurate for the sample sizes seen in online A/B tests (thousands+).

    Args:
        x_a: (n_a,) control metric per user. x_b: (n_b,) treatment.
    """
    raise NotImplementedError('TODO: implement welch_t_statistic (see the reference in src/mlbook)')

def sample_size_two_proportions(p0: float, mde_abs: float, alpha: float=0.05, power: float=0.8) -> int:
    """Users *per arm* to detect an absolute lift ``mde_abs`` on baseline ``p0``.

    ``n = (z_{1-alpha/2} sqrt(2 p (1-p)) + z_{power} sqrt(p0(1-p0) + p1(1-p1)))^2 / mde^2``
    with ``p1 = p0 + mde`` and ``p`` the average of ``p0, p1``.
    """
    raise NotImplementedError('TODO: implement sample_size_two_proportions (see the reference in src/mlbook)')

def bootstrap_ci(x: np.ndarray, stat, n_boot: int=2000, confidence: float=0.95, seed: int=0) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic ``stat: (N,) -> float``.

    Resample ``N`` points with replacement ``n_boot`` times, take the empirical
    ``(alpha/2, 1-alpha/2)`` quantiles of the statistic.
    """
    raise NotImplementedError('TODO: implement bootstrap_ci (see the reference in src/mlbook)')

def beta_binomial_posterior(a: float, b: float, successes: int, failures: int) -> tuple[float, float]:
    """Conjugate update: ``Beta(a, b)`` prior + binomial data -> ``Beta(a + s, b + f)``."""
    raise NotImplementedError('TODO: implement beta_binomial_posterior (see the reference in src/mlbook)')

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int=10) -> float:
    """ECE for binary predictions: ``sum_b (n_b / N) |acc_b - conf_b|``.

    Args:
        probs: (N,) predicted P(y=1). labels: (N,) in {0, 1}.
    """
    raise NotImplementedError('TODO: implement expected_calibration_error (see the reference in src/mlbook)')

def clt_sample_means(sampler, n_per_sample: int, n_samples: int) -> np.ndarray:
    """Draw ``n_samples`` means of ``n_per_sample`` draws each: the CLT experiment.

    Args:
        sampler: callable ``sampler(size) -> (size,)`` from any finite-variance law.
    Returns:
        (n_samples,) sample means; approximately Gaussian for large ``n_per_sample``.
    """
    raise NotImplementedError('TODO: implement clt_sample_means (see the reference in src/mlbook)')
