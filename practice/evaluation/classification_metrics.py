# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/evaluation/classification_metrics.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k classification_metrics -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py evaluation/classification_metrics --force

"""Classification metrics from scratch (NumPy): confusion matrix, P/R/F1 with
micro/macro averaging, ROC / PR curves and their areas, Brier score,
calibration curve bins, and cost-sensitive threshold selection.
"""
from __future__ import annotations
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """C[i, j] = number of examples with true class i predicted as j. -> (K, K)."""
    raise NotImplementedError('TODO: implement confusion_matrix (see the reference in src/mlbook)')

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, average: str='binary') -> tuple[float, float, float]:
    """P, R, F1. average in {"binary" (class 1), "micro", "macro"}.

    micro: pool TP/FP/FN over classes; macro: mean of per-class scores.
    """
    raise NotImplementedError('TODO: implement precision_recall_f1 (see the reference in src/mlbook)')

def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FPR, TPR and thresholds by sweeping the threshold over sorted scores.

    y_true (N,) in {0,1}, scores (N,) -> fpr (T+1,), tpr (T+1,), thresholds (T,).
    """
    raise NotImplementedError('TODO: implement roc_curve (see the reference in src/mlbook)')

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC as the Mann-Whitney rank statistic:
    P(score_pos > score_neg) + 0.5 P(tie). Equivalent to the trapezoid area."""
    raise NotImplementedError('TODO: implement roc_auc (see the reference in src/mlbook)')

def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at each distinct threshold, in decreasing-threshold order.
    -> precision (T,), recall (T,). Recall is non-decreasing along the arrays."""
    raise NotImplementedError('TODO: implement pr_curve (see the reference in src/mlbook)')

def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AP = sum_t (R_t - R_{t-1}) P_t  (step-wise area under the PR curve)."""
    raise NotImplementedError('TODO: implement average_precision (see the reference in src/mlbook)')

def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error of the probability of the positive class. probs (N,), y (N,)."""
    raise NotImplementedError('TODO: implement brier_score (see the reference in src/mlbook)')

def calibration_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int=10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width bins on [0,1]. -> (mean confidence (B,), empirical accuracy (B,), count (B,)).
    Empty bins get NaN."""
    raise NotImplementedError('TODO: implement calibration_curve (see the reference in src/mlbook)')

def best_threshold_for_cost(scores: np.ndarray, y_true: np.ndarray, cost_fp: float, cost_fn: float) -> tuple[float, float]:
    """Threshold minimising expected cost = cost_fp * FP + cost_fn * FN over all
    distinct thresholds. Returns (threshold, min_cost). Bayes rule: for calibrated
    p, predict positive iff p >= cost_fp / (cost_fp + cost_fn)."""
    raise NotImplementedError('TODO: implement best_threshold_for_cost (see the reference in src/mlbook)')
