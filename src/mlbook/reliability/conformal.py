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
    n = len(scores)
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float(np.inf)  # not enough calibration points for this alpha
    return float(np.quantile(scores, level, method="higher"))


class SplitConformalClassifier:
    """Prediction sets from softmax scores with s(x, y) = 1 - p_y(x)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.q_hat: float = np.inf

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalClassifier":
        """probs_cal (n, K), y_cal (n,)."""
        scores = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]  # (n,)
        self.q_hat = conformal_quantile(scores, self.alpha)
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """probs (N, K) -> boolean membership matrix (N, K)."""
        return (1.0 - probs) <= self.q_hat  # (N, K)


class SplitConformalRegressor:
    """Symmetric intervals mu(x) +/- q_hat with s = |y - mu(x)|."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.q_hat: float = np.inf

    def calibrate(self, mu_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalRegressor":
        scores = np.abs(y_cal - mu_cal)  # (n,)
        self.q_hat = conformal_quantile(scores, self.alpha)
        return self

    def predict_intervals(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mu (N,) -> (lower (N,), upper (N,))."""
        return mu - self.q_hat, mu + self.q_hat


def empirical_coverage_sets(sets: np.ndarray, y: np.ndarray) -> float:
    """sets (N, K) boolean, y (N,) -> fraction of rows whose set contains y."""
    return float(np.mean(sets[np.arange(len(y)), y]))


def empirical_coverage_intervals(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))
