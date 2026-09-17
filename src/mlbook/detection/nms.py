"""Non-maximum suppression: greedy, brute-force reference, class-aware (batched), soft-NMS."""

from __future__ import annotations

import numpy as np

from .boxes import box_iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Greedy NMS: repeatedly keep the highest-scoring box and drop every box with IoU > τ to it.

    Worst case ``O(N²)`` IoU evaluations (all boxes disjoint); in practice each kept box
    removes many.  ``N`` is the pre-NMS top-k, so pick that k with latency in mind.

    Args:
        boxes: (N, 4) xyxy.  scores: (N,).
    Returns:
        (K,) indices of kept boxes in descending score order.
    """
    order = np.argsort(-scores)  # (N,) highest first
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]  # (N_remaining,)
        order = order[1:][ious <= iou_threshold]  # (N_survivors,)
    return np.asarray(keep, dtype=np.int64)  # (K,)


def nms_bruteforce(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Reference: a box survives iff no *kept* higher-scoring box overlaps it above τ. (K,) indices."""
    order = np.argsort(-scores)  # (N,)
    iou = box_iou(boxes, boxes)  # (N, N)
    suppressed = np.zeros(len(boxes), dtype=bool)  # (N,)
    keep: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in order:
            if scores[j] < scores[i] and iou[i, j] > iou_threshold:
                suppressed[j] = True
    return np.asarray(keep, dtype=np.int64)  # (K,)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Class-aware NMS via the offset trick: shift each class's boxes to a disjoint region.

    Boxes of different classes then have IoU 0 and never suppress each other, so a single
    ``nms`` call replaces a Python loop over classes (what torchvision's ``batched_nms`` does).

    Args:
        boxes: (N, 4).  scores: (N,).  class_ids: (N,) ints.
    Returns:
        (K,) kept indices.
    """
    max_coord = boxes.max() + 1.0  # scalar larger than any coordinate
    offsets = class_ids.astype(np.float64)[:, None] * max_coord  # (N, 1)
    shifted = boxes + offsets  # (N, 4) each class in its own strip
    return nms(shifted, scores, iou_threshold)  # (K,)


def soft_nms(boxes: np.ndarray, scores: np.ndarray, sigma: float = 0.5, score_threshold: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS (Bodla et al. 2017), Gaussian variant: decay instead of delete.

    ``s_j ← s_j · exp(−IoU(M, b_j)² / σ)`` for every remaining box ``b_j`` after selecting ``M``;
    boxes whose score falls below ``score_threshold`` are dropped.  Helps crowded scenes
    where two true objects overlap heavily.

    Returns:
        (kept indices (K,), their decayed scores (K,)).
    """
    boxes = boxes.copy()
    scores = scores.astype(np.float64).copy()  # (N,)
    remaining = np.arange(len(boxes))  # (N,)
    keep: list[int] = []
    keep_scores: list[float] = []
    while remaining.size > 0:
        best = int(np.argmax(scores[remaining]))
        i = int(remaining[best])
        keep.append(i)
        keep_scores.append(float(scores[i]))
        remaining = np.delete(remaining, best)  # (N_remaining,)
        if remaining.size == 0:
            break
        ious = box_iou(boxes[i : i + 1], boxes[remaining])[0]  # (N_remaining,)
        scores[remaining] *= np.exp(-(ious**2) / sigma)  # Gaussian decay
        remaining = remaining[scores[remaining] >= score_threshold]
    return np.asarray(keep, dtype=np.int64), np.asarray(keep_scores)
