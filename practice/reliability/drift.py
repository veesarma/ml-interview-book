# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/reliability/drift.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k drift -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py reliability/drift --force

"""Drift detection (NumPy): two-sample KS test, population stability index,
classifier two-sample test (C2ST) with a from-scratch logistic regression, and
importance weights for covariate shift from the same classifier.
"""
from __future__ import annotations
import numpy as np

def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic sup_x |F_a(x) - F_b(x)|. a (n,), b (m,)."""
    raise NotImplementedError('TODO: implement ks_statistic (see the reference in src/mlbook)')

def ks_pvalue(stat: float, n: int, m: int, terms: int=100) -> float:
    """Asymptotic p-value: P(K > lambda) = 2 sum_{j>=1} (-1)^{j-1} exp(-2 j^2 lambda^2),
    lambda = stat * sqrt(nm / (n + m))."""
    raise NotImplementedError('TODO: implement ks_pvalue (see the reference in src/mlbook)')

def population_stability_index(ref: np.ndarray, cur: np.ndarray, n_bins: int=10, eps: float=1e-06) -> float:
    """PSI = sum_b (cur_b - ref_b) ln(cur_b / ref_b) on quantile bins of the reference."""
    raise NotImplementedError('TODO: implement population_stability_index (see the reference in src/mlbook)')

def _fit_logreg(X: np.ndarray, y: np.ndarray, lr: float=0.1, n_iters: int=500, l2: float=0.001) -> tuple[np.ndarray, float]:
    """Gradient-descent logistic regression. X (N, d), y (N,) -> (w (d,), b)."""
    raise NotImplementedError('TODO: implement _fit_logreg (see the reference in src/mlbook)')

def classifier_two_sample_test(X_ref: np.ndarray, X_cur: np.ndarray, test_frac: float=0.5, seed: int=0) -> dict[str, float]:
    """Train a classifier to tell reference from current; held-out accuracy far above 0.5
    means the distributions differ. Returns accuracy and a one-sided binomial z-test p-value."""
    raise NotImplementedError('TODO: implement classifier_two_sample_test (see the reference in src/mlbook)')

def importance_weights(X_ref: np.ndarray, X_cur: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Density-ratio weights w(x) = p_cur(x) / p_ref(x) = (n_ref / n_cur) * p(cur|x) / p(ref|x),
    estimated from a domain classifier. X_query (M, d) -> (M,)."""
    raise NotImplementedError('TODO: implement importance_weights (see the reference in src/mlbook)')

def _erf(x: float) -> float:
    """Abramowitz-Stegun 7.1.26 approximation of erf (|error| < 1.5e-7)."""
    raise NotImplementedError('TODO: implement _erf (see the reference in src/mlbook)')
