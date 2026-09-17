"""Drift detection (NumPy): two-sample KS test, population stability index,
classifier two-sample test (C2ST) with a from-scratch logistic regression, and
importance weights for covariate shift from the same classifier.
"""
from __future__ import annotations

import numpy as np


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic sup_x |F_a(x) - F_b(x)|. a (n,), b (m,)."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])  # (n+m,)
    Fa = np.searchsorted(a, grid, side="right") / len(a)  # (n+m,)
    Fb = np.searchsorted(b, grid, side="right") / len(b)  # (n+m,)
    return float(np.max(np.abs(Fa - Fb)))


def ks_pvalue(stat: float, n: int, m: int, terms: int = 100) -> float:
    """Asymptotic p-value: P(K > lambda) = 2 sum_{j>=1} (-1)^{j-1} exp(-2 j^2 lambda^2),
    lambda = stat * sqrt(nm / (n + m))."""
    lam = stat * np.sqrt(n * m / (n + m))
    j = np.arange(1, terms + 1)  # (terms,)
    p = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j**2 * lam**2))
    return float(np.clip(p, 0.0, 1.0))


def population_stability_index(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """PSI = sum_b (cur_b - ref_b) ln(cur_b / ref_b) on quantile bins of the reference."""
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))  # (B+1,)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps  # (B,)
    c = np.histogram(cur, edges)[0] / len(cur) + eps  # (B,)
    return float(np.sum((c - r) * np.log(c / r)))


def _fit_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iters: int = 500, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    """Gradient-descent logistic regression. X (N, d), y (N,) -> (w (d,), b)."""
    N, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(n_iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))  # (N,)
        g = p - y  # (N,)  dL/dz for cross-entropy
        w -= lr * (X.T @ g / N + l2 * w)
        b -= lr * float(g.mean())
    return w, b


def classifier_two_sample_test(
    X_ref: np.ndarray, X_cur: np.ndarray, test_frac: float = 0.5, seed: int = 0
) -> dict[str, float]:
    """Train a classifier to tell reference from current; held-out accuracy far above 0.5
    means the distributions differ. Returns accuracy and a one-sided binomial z-test p-value."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([X_ref, X_cur])  # (N, d)
    y = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_cur))])  # (N,)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - mu) / sd  # (N, d) standardise
    perm = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    te, tr = perm[:n_test], perm[n_test:]
    w, b = _fit_logreg(X[tr], y[tr])
    acc = float(np.mean(((X[te] @ w + b) > 0) == y[te]))
    # H0: accuracy = 0.5. z = (acc - 0.5) / sqrt(0.25 / n_test); p = P(Z > z).
    z = (acc - 0.5) / np.sqrt(0.25 / n_test)
    p = 0.5 * (1.0 - _erf(z / np.sqrt(2.0)))
    return {"accuracy": acc, "z": float(z), "p_value": float(p)}


def importance_weights(X_ref: np.ndarray, X_cur: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Density-ratio weights w(x) = p_cur(x) / p_ref(x) = (n_ref / n_cur) * p(cur|x) / p(ref|x),
    estimated from a domain classifier. X_query (M, d) -> (M,)."""
    X = np.concatenate([X_ref, X_cur])
    y = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_cur))])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    w, b = _fit_logreg((X - mu) / sd, y)
    p = 1.0 / (1.0 + np.exp(-(((X_query - mu) / sd) @ w + b)))  # (M,) P(cur | x)
    return (len(X_ref) / len(X_cur)) * p / np.maximum(1.0 - p, 1e-8)  # (M,)


def _erf(x: float) -> float:
    """Abramowitz-Stegun 7.1.26 approximation of erf (|error| < 1.5e-7)."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))
