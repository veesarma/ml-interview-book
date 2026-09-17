"""Classification metrics from scratch (NumPy): confusion matrix, P/R/F1 with
micro/macro averaging, ROC / PR curves and their areas, Brier score,
calibration curve bins, and cost-sensitive threshold selection.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """C[i, j] = number of examples with true class i predicted as j. -> (K, K)."""
    C = np.zeros((n_classes, n_classes), dtype=int)  # (K, K)
    np.add.at(C, (y_true.astype(int), y_pred.astype(int)), 1)
    return C


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, average: str = "binary"
) -> tuple[float, float, float]:
    """P, R, F1. average in {"binary" (class 1), "micro", "macro"}.

    micro: pool TP/FP/FN over classes; macro: mean of per-class scores.
    """
    C = confusion_matrix(y_true, y_pred, n_classes)  # (K, K)
    tp = np.diag(C).astype(float)  # (K,)
    fp = C.sum(axis=0) - tp  # (K,) predicted as k but not k
    fn = C.sum(axis=1) - tp  # (K,) truly k but predicted otherwise
    if average == "binary":
        tp, fp, fn = tp[1:2], fp[1:2], fn[1:2]
    elif average == "micro":
        tp, fp, fn = tp.sum(keepdims=True), fp.sum(keepdims=True), fn.sum(keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)  # (K,) or (1,)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(p.mean()), float(r.mean()), float(f1.mean())


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FPR, TPR and thresholds by sweeping the threshold over sorted scores.

    y_true (N,) in {0,1}, scores (N,) -> fpr (T+1,), tpr (T+1,), thresholds (T,).
    """
    order = np.argsort(-scores, kind="stable")  # (N,)
    y = y_true[order].astype(float)  # (N,)
    s = scores[order]  # (N,)
    tps = np.cumsum(y)  # (N,) true positives if we predict positive for top i
    fps = np.cumsum(1 - y)  # (N,)
    # keep the last index of each distinct score (ties handled together)
    distinct = np.flatnonzero(np.r_[np.diff(s) != 0, True])  # (T,)
    tpr = np.r_[0.0, tps[distinct] / max(y.sum(), 1)]  # (T+1,)
    fpr = np.r_[0.0, fps[distinct] / max((1 - y).sum(), 1)]  # (T+1,)
    return fpr, tpr, s[distinct]


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC as the Mann-Whitney rank statistic:
    P(score_pos > score_neg) + 0.5 P(tie). Equivalent to the trapezoid area."""
    pos = scores[y_true == 1]  # (P,)
    neg = scores[y_true == 0]  # (Q,)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]  # (P, Q)
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at each distinct threshold, in decreasing-threshold order.
    -> precision (T,), recall (T,). Recall is non-decreasing along the arrays."""
    order = np.argsort(-scores, kind="stable")
    y = y_true[order].astype(float)  # (N,)
    s = scores[order]
    tps = np.cumsum(y)  # (N,)
    n_pred = np.arange(1, len(y) + 1)  # (N,)
    distinct = np.flatnonzero(np.r_[np.diff(s) != 0, True])  # (T,)
    precision = tps[distinct] / n_pred[distinct]  # (T,)
    recall = tps[distinct] / max(y.sum(), 1)  # (T,)
    return precision, recall


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AP = sum_t (R_t - R_{t-1}) P_t  (step-wise area under the PR curve)."""
    precision, recall = pr_curve(y_true, scores)
    prev_r = np.r_[0.0, recall[:-1]]  # (T,)
    return float(np.sum((recall - prev_r) * precision))


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error of the probability of the positive class. probs (N,), y (N,)."""
    return float(np.mean((probs - y_true) ** 2))


def calibration_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width bins on [0,1]. -> (mean confidence (B,), empirical accuracy (B,), count (B,)).
    Empty bins get NaN."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)  # (B+1,)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)  # (N,)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        count[b] = m.sum()
        if count[b]:
            conf[b] = probs[m].mean()
            acc[b] = y_true[m].mean()
    return conf, acc, count


def best_threshold_for_cost(
    scores: np.ndarray, y_true: np.ndarray, cost_fp: float, cost_fn: float
) -> tuple[float, float]:
    """Threshold minimising expected cost = cost_fp * FP + cost_fn * FN over all
    distinct thresholds. Returns (threshold, min_cost). Bayes rule: for calibrated
    p, predict positive iff p >= cost_fp / (cost_fp + cost_fn)."""
    best_t, best_c = float("inf"), float("inf")
    for t in np.unique(scores):
        pred = scores >= t  # (N,)
        fp = np.sum(pred & (y_true == 0))
        fn = np.sum(~pred & (y_true == 1))
        c = cost_fp * fp + cost_fn * fn
        if c < best_c:
            best_t, best_c = float(t), float(c)
    return best_t, best_c
