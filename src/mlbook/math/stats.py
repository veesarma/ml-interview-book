"""Statistics: MLE/MAP, bias-variance, intervals, A/B tests, bootstrap, calibration.

Pure NumPy; the normal CDF comes from ``mlbook.math.probability``.
"""

from __future__ import annotations

import math

import numpy as np

from mlbook.math.probability import normal_cdf

# ---------------------------------------------------------------------------
# Point estimation: MLE and MAP
# ---------------------------------------------------------------------------


def mle_gaussian(x: np.ndarray) -> tuple[float, float]:
    """MLE of a univariate Gaussian: ``mu = mean(x)``, ``sigma^2 = mean((x-mu)^2)`` (biased, 1/N).

    Args:
        x: (N,).
    """
    mu = float(x.mean())
    var = float(np.mean((x - mu) ** 2))  # divides by N, not N-1
    return mu, var


def map_gaussian_mean(x: np.ndarray, sigma: float, mu0: float, tau: float) -> float:
    """MAP of the mean with likelihood ``N(x; mu, sigma^2)`` and prior ``N(mu; mu0, tau^2)``.

    ``mu_MAP = (sum x / sigma^2 + mu0 / tau^2) / (N / sigma^2 + 1 / tau^2)``:
    a precision-weighted average of the data mean and the prior mean.

    Args:
        x: (N,).
    """
    n = x.shape[0]
    precision_data = n / sigma**2
    precision_prior = 1.0 / tau**2
    return float((x.sum() / sigma**2 + mu0 * precision_prior) / (precision_data + precision_prior))


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge = MAP with Gaussian prior: ``w = (X^T X + lam I)^{-1} X^T y``.

    Args:
        X: (N, d). y: (N,). lam: L2 strength = sigma^2 / tau^2 in the Bayesian reading.
    Returns:
        (d,).
    """
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)  # (d,)


# ---------------------------------------------------------------------------
# Bias-variance decomposition (by simulation)
# ---------------------------------------------------------------------------


def bias_variance_decomposition(predictions: np.ndarray, y_true: np.ndarray, noise_var: float = 0.0) -> dict:
    """Decompose expected squared error over ``M`` re-trained models.

    ``E[(f_hat - y)^2] = (E f_hat - f)^2 + Var[f_hat] + sigma^2``.

    Args:
        predictions: (M, N) predictions of M independently trained models on N test points.
        y_true: (N,) noiseless targets ``f(x)``.
    Returns:
        dict with mean ``bias_sq``, ``variance``, ``noise``, ``total`` over the N points.
    """
    mean_pred = predictions.mean(axis=0)  # (N,) E[f_hat(x)]
    bias_sq = np.mean((mean_pred - y_true) ** 2)  # scalar
    variance = np.mean(predictions.var(axis=0))  # scalar: mean over x of Var[f_hat(x)]
    total = bias_sq + variance + noise_var
    return {"bias_sq": float(bias_sq), "variance": float(variance), "noise": noise_var, "total": float(total)}


# ---------------------------------------------------------------------------
# Intervals, tests, bootstrap
# ---------------------------------------------------------------------------


def z_critical(confidence: float) -> float:
    """Two-sided standard-normal critical value, e.g. 1.96 for 0.95 (bisection on Phi)."""
    target = 1.0 - (1.0 - confidence) / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if normal_cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def confidence_interval_mean(x: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation CI for a mean: ``mean +- z * s / sqrt(N)`` (CLT).

    Args:
        x: (N,), N large enough for the CLT (rule of thumb: > 30, non-heavy-tailed).
    """
    n = x.shape[0]
    se = x.std(ddof=1) / math.sqrt(n)
    z = z_critical(confidence)
    return float(x.mean() - z * se), float(x.mean() + z * se)


def two_proportion_z_test(k_a: int, n_a: int, k_b: int, n_b: int) -> dict:
    """A/B test on conversion rates: pooled two-proportion z-test.

    ``z = (p_b - p_a) / sqrt(p (1-p) (1/n_a + 1/n_b))``, ``p`` pooled.
    Two-sided p-value ``2 (1 - Phi(|z|))``.

    Args:
        k_a, n_a: conversions and users in control; k_b, n_b: in treatment.
    Returns:
        dict with lift ``p_b - p_a``, ``z``, ``p_value``, ``se``.
    """
    p_a, p_b = k_a / n_a, k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2.0 * (1.0 - float(normal_cdf(abs(z))))
    return {"lift": p_b - p_a, "z": z, "p_value": p_value, "se": se}


def welch_t_statistic(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    """Welch's t for unequal variances: ``t = (mean_b - mean_a) / sqrt(s_a^2/n_a + s_b^2/n_b)``.

    The p-value uses the normal approximation to the t-distribution, which is
    accurate for the sample sizes seen in online A/B tests (thousands+).

    Args:
        x_a: (n_a,) control metric per user. x_b: (n_b,) treatment.
    """
    n_a, n_b = x_a.shape[0], x_b.shape[0]
    var_a, var_b = x_a.var(ddof=1), x_b.var(ddof=1)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    t = (x_b.mean() - x_a.mean()) / se
    dof = (var_a / n_a + var_b / n_b) ** 2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
    p_value = 2.0 * (1.0 - float(normal_cdf(abs(t))))
    return {"t": float(t), "dof": float(dof), "p_value": p_value, "se": se}


def sample_size_two_proportions(p0: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users *per arm* to detect an absolute lift ``mde_abs`` on baseline ``p0``.

    ``n = (z_{1-alpha/2} sqrt(2 p (1-p)) + z_{power} sqrt(p0(1-p0) + p1(1-p1)))^2 / mde^2``
    with ``p1 = p0 + mde`` and ``p`` the average of ``p0, p1``.
    """
    p1 = p0 + mde_abs
    p_bar = 0.5 * (p0 + p1)
    z_alpha = z_critical(1 - alpha)  # two-sided
    z_beta = z_critical(1 - 2 * (1 - power))  # one-sided quantile at `power`
    num = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return int(math.ceil(num / mde_abs**2))


def bootstrap_ci(x: np.ndarray, stat, n_boot: int = 2000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic ``stat: (N,) -> float``.

    Resample ``N`` points with replacement ``n_boot`` times, take the empirical
    ``(alpha/2, 1-alpha/2)`` quantiles of the statistic.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))  # (n_boot, N) resample indices
    stats = np.array([stat(x[row]) for row in idx])  # (n_boot,)
    alpha = 1 - confidence
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Bayesian updating and calibration
# ---------------------------------------------------------------------------


def beta_binomial_posterior(a: float, b: float, successes: int, failures: int) -> tuple[float, float]:
    """Conjugate update: ``Beta(a, b)`` prior + binomial data -> ``Beta(a + s, b + f)``."""
    return a + successes, b + failures


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE for binary predictions: ``sum_b (n_b / N) |acc_b - conf_b|``.

    Args:
        probs: (N,) predicted P(y=1). labels: (N,) in {0, 1}.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)  # (n_bins+1,)
    ece = 0.0
    n = probs.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi)  # (N,) bool
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def clt_sample_means(sampler, n_per_sample: int, n_samples: int) -> np.ndarray:
    """Draw ``n_samples`` means of ``n_per_sample`` draws each: the CLT experiment.

    Args:
        sampler: callable ``sampler(size) -> (size,)`` from any finite-variance law.
    Returns:
        (n_samples,) sample means; approximately Gaussian for large ``n_per_sample``.
    """
    draws = sampler((n_samples, n_per_sample))  # (n_samples, n_per_sample)
    return draws.mean(axis=1)  # (n_samples,)
