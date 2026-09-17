# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/detection/anchors.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k anchors -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py detection/anchors --force

"""Anchor generation and anchor-to-ground-truth assignment (NumPy).

An anchor is a reference box placed at every feature-map cell; the detector predicts a
class score and a Δ-offset (see ``boxes.encode_boxes``) *per anchor*.
"""
from __future__ import annotations
import numpy as np
from .boxes import box_iou, xyxy_to_cxcywh

def base_anchors(size: float, ratios: tuple[float, ...]=(0.5, 1.0, 2.0), scales: tuple[float, ...]=(1.0, 2 ** (1 / 3), 2 ** (2 / 3))) -> np.ndarray:
    """Anchor templates centred at the origin: every (ratio, scale) pair, area ``(size·scale)²``.

    Aspect ratio ``r = h / w`` with fixed area gives ``w = size·scale / √r``, ``h = w · r``.
    Returns:
        (A, 4) xyxy with A = len(ratios) · len(scales).
    """
    raise NotImplementedError('TODO: implement base_anchors (see the reference in src/mlbook)')

def grid_anchors(feat_h: int, feat_w: int, stride: int, templates: np.ndarray) -> np.ndarray:
    """Tile ``templates`` over a ``feat_h × feat_w`` feature map with the given stride.

    The centre of cell ``(i, j)`` is at image coordinates ``((j + 0.5)·stride, (i + 0.5)·stride)``.
    Returns:
        (feat_h · feat_w · A, 4) xyxy anchors, ordered (i, j, a) row-major.
    """
    raise NotImplementedError('TODO: implement grid_anchors (see the reference in src/mlbook)')

def multilevel_anchors(image_hw: tuple[int, int], strides: tuple[int, ...]=(8, 16, 32, 64, 128), size_per_stride: float=4.0) -> np.ndarray:
    """RetinaNet-style anchors for FPN levels P3–P7: anchor size = ``size_per_stride · stride``.

    Returns:
        (N_total, 4) concatenated over levels (finest level first).
    """
    raise NotImplementedError('TODO: implement multilevel_anchors (see the reference in src/mlbook)')

def assign_max_iou(anchors: np.ndarray, gt_boxes: np.ndarray, pos_iou: float=0.5, neg_iou: float=0.4) -> tuple[np.ndarray, np.ndarray]:
    """Classic IoU-threshold assignment (RetinaNet / Faster R-CNN RPN).

    * anchor with max-IoU ≥ ``pos_iou`` → positive, matched to its argmax GT;
    * max-IoU < ``neg_iou`` → negative (background);
    * in between → ignored (label −1);
    * every GT additionally claims its own best anchor, so no GT is left unmatched.

    Args:
        anchors: (N, 4).  gt_boxes: (M, 4).
    Returns:
        labels (N,) ∈ {1 pos, 0 neg, −1 ignore};  matched_gt (N,) index into gt_boxes (valid where pos).
    """
    raise NotImplementedError('TODO: implement assign_max_iou (see the reference in src/mlbook)')

def assign_atss(anchors: np.ndarray, gt_boxes: np.ndarray, level_ids: np.ndarray, top_k: int=9) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive Training Sample Selection (Zhang et al. 2020), simplified.

    For each GT: (1) take the ``top_k`` closest anchors (centre distance) *per level*;
    (2) compute IoU of those candidates; (3) threshold = mean + std of those IoUs;
    (4) positives = candidates with IoU ≥ threshold whose centre lies inside the GT.
    The per-GT adaptive threshold replaces the fixed 0.5/0.4, so small or thin objects
    (whose best IoUs are low) still get positives.

    Args:
        anchors: (N, 4).  gt_boxes: (M, 4).  level_ids: (N,) FPN level index per anchor.
    Returns:
        labels (N,) ∈ {1, 0};  matched_gt (N,).
    """
    raise NotImplementedError('TODO: implement assign_atss (see the reference in src/mlbook)')
