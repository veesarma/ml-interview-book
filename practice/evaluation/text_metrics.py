# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/evaluation/text_metrics.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k text_metrics -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py evaluation/text_metrics --force

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
    raise NotImplementedError('TODO: implement edit_distance (see the reference in src/mlbook)')

def cer(ref: str, hyp: str) -> float:
    """Character error rate = (S + D + I) / len(ref)."""
    raise NotImplementedError('TODO: implement cer (see the reference in src/mlbook)')

def wer(ref: str, hyp: str) -> float:
    """Word error rate on whitespace tokens."""
    raise NotImplementedError('TODO: implement wer (see the reference in src/mlbook)')

def text_spotting_f1(pred_boxes: np.ndarray, pred_texts: list[str], gt_boxes: np.ndarray, gt_texts: list[str], iou_thr: float=0.5) -> tuple[float, float, float]:
    """End-to-end text spotting: a prediction is a TP iff it matches an unused GT
    with IoU >= thr AND the transcription is identical (case-insensitive).
    pred_boxes (P, 4), gt_boxes (G, 4). Returns (precision, recall, f1)."""
    raise NotImplementedError('TODO: implement text_spotting_f1 (see the reference in src/mlbook)')
