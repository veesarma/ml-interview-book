"""Anchor generation and anchor-to-ground-truth assignment (NumPy).

An anchor is a reference box placed at every feature-map cell; the detector predicts a
class score and a Δ-offset (see ``boxes.encode_boxes``) *per anchor*.
"""

from __future__ import annotations

import numpy as np

from .boxes import box_iou, xyxy_to_cxcywh


def base_anchors(size: float, ratios: tuple[float, ...] = (0.5, 1.0, 2.0), scales: tuple[float, ...] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3))) -> np.ndarray:
    """Anchor templates centred at the origin: every (ratio, scale) pair, area ``(size·scale)²``.

    Aspect ratio ``r = h / w`` with fixed area gives ``w = size·scale / √r``, ``h = w · r``.
    Returns:
        (A, 4) xyxy with A = len(ratios) · len(scales).
    """
    out = []
    for r in ratios:
        for s in scales:
            w = size * s / np.sqrt(r)
            h = w * r
            out.append([-w / 2, -h / 2, w / 2, h / 2])
    return np.asarray(out, dtype=np.float64)  # (A, 4)


def grid_anchors(feat_h: int, feat_w: int, stride: int, templates: np.ndarray) -> np.ndarray:
    """Tile ``templates`` over a ``feat_h × feat_w`` feature map with the given stride.

    The centre of cell ``(i, j)`` is at image coordinates ``((j + 0.5)·stride, (i + 0.5)·stride)``.
    Returns:
        (feat_h · feat_w · A, 4) xyxy anchors, ordered (i, j, a) row-major.
    """
    cy = (np.arange(feat_h) + 0.5) * stride  # (H_f,)
    cx = (np.arange(feat_w) + 0.5) * stride  # (W_f,)
    grid_y, grid_x = np.meshgrid(cy, cx, indexing="ij")  # (H_f, W_f) each
    shifts = np.stack([grid_x, grid_y, grid_x, grid_y], axis=-1).reshape(-1, 1, 4)  # (H_f·W_f, 1, 4)
    anchors = templates[None, :, :] + shifts  # (H_f·W_f, A, 4)
    return anchors.reshape(-1, 4)  # (H_f·W_f·A, 4)


def multilevel_anchors(image_hw: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32, 64, 128), size_per_stride: float = 4.0) -> np.ndarray:
    """RetinaNet-style anchors for FPN levels P3–P7: anchor size = ``size_per_stride · stride``.

    Returns:
        (N_total, 4) concatenated over levels (finest level first).
    """
    H, W = image_hw
    per_level = []
    for stride in strides:
        feat_h, feat_w = int(np.ceil(H / stride)), int(np.ceil(W / stride))
        per_level.append(grid_anchors(feat_h, feat_w, stride, base_anchors(size_per_stride * stride)))
    return np.concatenate(per_level, axis=0)  # (Σ_l H_l W_l A, 4)


def assign_max_iou(anchors: np.ndarray, gt_boxes: np.ndarray, pos_iou: float = 0.5, neg_iou: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
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
    N = len(anchors)
    if len(gt_boxes) == 0:
        return np.zeros(N, dtype=np.int64), np.full(N, -1, dtype=np.int64)
    iou = box_iou(anchors, gt_boxes)  # (N, M)
    matched_gt = iou.argmax(axis=1)  # (N,) best GT per anchor
    max_iou = iou[np.arange(N), matched_gt]  # (N,)
    labels = np.full(N, -1, dtype=np.int64)  # (N,) start as ignore
    labels[max_iou < neg_iou] = 0
    labels[max_iou >= pos_iou] = 1
    best_anchor_per_gt = iou.argmax(axis=0)  # (M,)  low-quality matches: each GT keeps its best anchor
    labels[best_anchor_per_gt] = 1
    matched_gt[best_anchor_per_gt] = np.arange(len(gt_boxes))
    return labels, matched_gt


def assign_atss(anchors: np.ndarray, gt_boxes: np.ndarray, level_ids: np.ndarray, top_k: int = 9) -> tuple[np.ndarray, np.ndarray]:
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
    N, M = len(anchors), len(gt_boxes)
    labels = np.zeros(N, dtype=np.int64)  # (N,)
    matched_gt = np.full(N, -1, dtype=np.int64)  # (N,)
    if M == 0:
        return labels, matched_gt
    iou = box_iou(anchors, gt_boxes)  # (N, M)
    a_c = xyxy_to_cxcywh(anchors)[:, :2]  # (N, 2)
    g_c = xyxy_to_cxcywh(gt_boxes)[:, :2]  # (M, 2)
    dist = np.linalg.norm(a_c[:, None, :] - g_c[None, :, :], axis=-1)  # (N, M)
    best_iou_for_anchor = np.full(N, -1.0)  # (N,) resolve anchors claimed by several GTs
    for m in range(M):
        candidates = []
        for level in np.unique(level_ids):
            idx = np.flatnonzero(level_ids == level)  # anchors on this level
            k = min(top_k, len(idx))
            closest = idx[np.argsort(dist[idx, m])[:k]]  # (k,)
            candidates.append(closest)
        cand = np.concatenate(candidates)  # (L·k,)
        cand_iou = iou[cand, m]  # (L·k,)
        threshold = cand_iou.mean() + cand_iou.std()
        inside = (a_c[cand, 0] > gt_boxes[m, 0]) & (a_c[cand, 0] < gt_boxes[m, 2]) & (a_c[cand, 1] > gt_boxes[m, 1]) & (a_c[cand, 1] < gt_boxes[m, 3])  # (L·k,)
        pos = cand[(cand_iou >= threshold) & inside]  # (P,)
        better = iou[pos, m] > best_iou_for_anchor[pos]
        pos = pos[better]
        labels[pos] = 1
        matched_gt[pos] = m
        best_iou_for_anchor[pos] = iou[pos, m]
    return labels, matched_gt
