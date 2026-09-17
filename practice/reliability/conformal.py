# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/reliability/conformal.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k conformal -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py reliability/conformal --force

"""Split conformal prediction (NumPy).

Given a conformity score s(x, y) (large = worse fit), calibration scores
s_1..s_n, and a miscoverage level alpha, set
    q_hat = the ceil((n + 1)(1 - alpha)) / n  empirical quantile of the scores
and predict C(x) = { y : s(x, y) <= q_hat }. Under exchangeability
    1 - alpha <= P(Y_test in C(X_test)) <= 1 - alpha + 1/(n + 1).
"""
from __future__ import annotations
import numpy as np

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected quantile of calibration scores (n,)."""
    raise NotImplementedError('TODO: implement conformal_quantile (see the reference in src/mlbook)')

class SplitConformalClassifier:
    """Prediction sets from softmax scores with s(x, y) = 1 - p_y(x)."""

    def __init__(self, alpha: float=0.1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> 'SplitConformalClassifier':
        """probs_cal (n, K), y_cal (n,)."""
        raise NotImplementedError('TODO: implement calibrate (see the reference in src/mlbook)')

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """probs (N, K) -> boolean membership matrix (N, K)."""
        raise NotImplementedError('TODO: implement predict_sets (see the reference in src/mlbook)')

class SplitConformalRegressor:
    """Symmetric intervals mu(x) +/- q_hat with s = |y - mu(x)|."""

    def __init__(self, alpha: float=0.1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def calibrate(self, mu_cal: np.ndarray, y_cal: np.ndarray) -> 'SplitConformalRegressor':
        raise NotImplementedError('TODO: implement calibrate (see the reference in src/mlbook)')

    def predict_intervals(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mu (N,) -> (lower (N,), upper (N,))."""
        raise NotImplementedError('TODO: implement predict_intervals (see the reference in src/mlbook)')

def empirical_coverage_sets(sets: np.ndarray, y: np.ndarray) -> float:
    """sets (N, K) boolean, y (N,) -> fraction of rows whose set contains y."""
    raise NotImplementedError('TODO: implement empirical_coverage_sets (see the reference in src/mlbook)')

def empirical_coverage_intervals(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError('TODO: implement empirical_coverage_intervals (see the reference in src/mlbook)')
