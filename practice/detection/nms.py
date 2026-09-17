# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/detection/nms.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k nms -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py detection/nms --force

"""Non-maximum suppression: greedy, brute-force reference, class-aware (batched), soft-NMS."""
from __future__ import annotations
import numpy as np
from .boxes import box_iou

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float=0.5) -> np.ndarray:
    """Greedy NMS: repeatedly keep the highest-scoring box and drop every box with IoU > τ to it.

    Worst case ``O(N²)`` IoU evaluations (all boxes disjoint); in practice each kept box
    removes many.  ``N`` is the pre-NMS top-k, so pick that k with latency in mind.

    Args:
        boxes: (N, 4) xyxy.  scores: (N,).
    Returns:
        (K,) indices of kept boxes in descending score order.
    """
    raise NotImplementedError('TODO: implement nms (see the reference in src/mlbook)')

def nms_bruteforce(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float=0.5) -> np.ndarray:
    """Reference: a box survives iff no *kept* higher-scoring box overlaps it above τ. (K,) indices."""
    raise NotImplementedError('TODO: implement nms_bruteforce (see the reference in src/mlbook)')

def batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float=0.5) -> np.ndarray:
    """Class-aware NMS via the offset trick: shift each class's boxes to a disjoint region.

    Boxes of different classes then have IoU 0 and never suppress each other, so a single
    ``nms`` call replaces a Python loop over classes (what torchvision's ``batched_nms`` does).

    Args:
        boxes: (N, 4).  scores: (N,).  class_ids: (N,) ints.
    Returns:
        (K,) kept indices.
    """
    raise NotImplementedError('TODO: implement batched_nms (see the reference in src/mlbook)')

def soft_nms(boxes: np.ndarray, scores: np.ndarray, sigma: float=0.5, score_threshold: float=0.001) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS (Bodla et al. 2017), Gaussian variant: decay instead of delete.

    ``s_j ← s_j · exp(−IoU(M, b_j)² / σ)`` for every remaining box ``b_j`` after selecting ``M``;
    boxes whose score falls below ``score_threshold`` are dropped.  Helps crowded scenes
    where two true objects overlap heavily.

    Returns:
        (kept indices (K,), their decayed scores (K,)).
    """
    raise NotImplementedError('TODO: implement soft_nms (see the reference in src/mlbook)')
