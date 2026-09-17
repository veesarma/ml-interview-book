"""OCR / text metrics: Levenshtein edit distance, CER, WER, and an end-to-end
text-spotting F-score (box IoU match + exact transcription).
"""
from __future__ import annotations

import numpy as np

from mlbook.evaluation.detection_map import iou_matrix


def edit_distance(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Levenshtein distance by dynamic programming with backtrace.

    D[i, j] = min(D[i-1, j] + 1 (deletion), D[i, j-1] + 1 (insertion),
                  D[i-1, j-1] + [ref_i != hyp_j] (substitution)).
    Returns (distance, substitutions, deletions, insertions).
    """
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=int)  # (n+1, m+1)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1])
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, sub)
    # backtrace to count operation types
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            Dl += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return int(D[n, m]), S, Dl, I


def cer(ref: str, hyp: str) -> float:
    """Character error rate = (S + D + I) / len(ref)."""
    d, _, _, _ = edit_distance(list(ref), list(hyp))
    return d / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    """Word error rate on whitespace tokens."""
    d, _, _, _ = edit_distance(ref.split(), hyp.split())
    return d / max(len(ref.split()), 1)


def text_spotting_f1(
    pred_boxes: np.ndarray, pred_texts: list[str],
    gt_boxes: np.ndarray, gt_texts: list[str], iou_thr: float = 0.5,
) -> tuple[float, float, float]:
    """End-to-end text spotting: a prediction is a TP iff it matches an unused GT
    with IoU >= thr AND the transcription is identical (case-insensitive).
    pred_boxes (P, 4), gt_boxes (G, 4). Returns (precision, recall, f1)."""
    P, G = len(pred_texts), len(gt_texts)
    if P == 0 or G == 0:
        return 0.0, 0.0, 0.0
    ious = iou_matrix(pred_boxes, gt_boxes)  # (P, G)
    used = np.zeros(G, dtype=bool)
    tp = 0
    for i in range(P):
        order = np.argsort(-ious[i])  # (G,)
        for j in order:
            if ious[i, j] < iou_thr:
                break
            if not used[j] and pred_texts[i].lower() == gt_texts[j].lower():
                used[j] = True
                tp += 1
                break
    prec, rec = tp / P, tp / G
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1
