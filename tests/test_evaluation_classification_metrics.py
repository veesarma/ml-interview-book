import numpy as np
from scipy import stats

from mlbook.evaluation import classification_metrics as cm

Y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1])


def test_confusion_matrix():
    C = cm.confusion_matrix(np.array([0, 1, 2, 2]), np.array([0, 2, 2, 1]), 3)
    assert np.array_equal(C, [[1, 0, 0], [0, 0, 1], [0, 1, 1]])


def test_precision_recall_f1_binary_and_averaging():
    y, p = np.array([1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1])
    P, R, F = cm.precision_recall_f1(y, p, 2, "binary")
    assert np.isclose(P, 2 / 3) and np.isclose(R, 2 / 3) and np.isclose(F, 2 / 3)
    y3, p3 = np.array([0, 0, 0, 1, 2]), np.array([0, 0, 1, 1, 0])
    Pm, Rm, Fm = cm.precision_recall_f1(y3, p3, 3, "micro")
    assert np.isclose(Pm, 3 / 5) and np.isclose(Rm, 3 / 5) and np.isclose(Fm, 3 / 5)  # micro = accuracy
    PM, RM, _ = cm.precision_recall_f1(y3, p3, 3, "macro")
    assert np.isclose(PM, (2 / 3 + 1 / 2 + 0) / 3) and np.isclose(RM, (2 / 3 + 1 + 0) / 3)


def test_roc_curve_and_auc_match_rank_statistic():
    fpr, tpr, _ = cm.roc_curve(Y, S)
    area = np.trapezoid(tpr, fpr)
    auc = cm.roc_auc(Y, S)
    assert np.isclose(area, auc)
    # Mann-Whitney U / (P*Q)
    u = stats.mannwhitneyu(S[Y == 1], S[Y == 0], alternative="two-sided").statistic
    assert np.isclose(auc, u / (4 * 4))


def test_pr_curve_and_average_precision_hand_computed():
    precision, recall = cm.pr_curve(Y, S)
    # hits at ranks 1,2,5,7 -> precisions 1, 1, 3/5, 4/7 ; each contributes 1/4 recall
    ap = cm.average_precision(Y, S)
    assert np.isclose(ap, (1 + 1 + 3 / 5 + 4 / 7) / 4)
    assert np.all(np.diff(recall) >= 0)


def test_brier_and_calibration_curve():
    p = np.array([0.9, 0.1, 0.8, 0.3])
    y = np.array([1, 0, 0, 1])
    assert np.isclose(cm.brier_score(p, y), np.mean([0.01, 0.01, 0.64, 0.49]))
    conf, acc, count = cm.calibration_curve(p, y, n_bins=2)
    assert count.tolist() == [2, 2] and np.isclose(acc[1], 0.5) and np.isclose(conf[0], 0.2)


def test_best_threshold_for_cost():
    t, c = cm.best_threshold_for_cost(S, Y, cost_fp=1.0, cost_fn=10.0)
    assert t <= 0.3  # expensive FNs push the threshold down until all positives caught
    assert c == 1.0 * 3  # three negatives above 0.3
