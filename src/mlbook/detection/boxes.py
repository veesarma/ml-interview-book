"""Box utilities (NumPy): IoU / GIoU / DIoU / CIoU, format conversion, Δ-encoding.

Boxes are ``(N, 4)`` float arrays in ``xyxy`` = ``(x1, y1, x2, y2)`` with ``x2 > x1``,
``y2 > y1``, continuous coordinates (no ``+1`` pixel convention).
"""

from __future__ import annotations

import numpy as np


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) xyxy → (N, 4) centre-size ``(cx, cy, w, h)``."""
    w = boxes[:, 2] - boxes[:, 0]  # (N,)
    h = boxes[:, 3] - boxes[:, 1]  # (N,)
    cx = boxes[:, 0] + 0.5 * w  # (N,)
    cy = boxes[:, 1] + 0.5 * h  # (N,)
    return np.stack([cx, cy, w, h], axis=1)  # (N, 4)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) centre-size → (N, 4) xyxy."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]  # each (N,)
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1)  # (N, 4)


def box_area(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) xyxy → (N,) areas."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between every box in ``a`` and every box in ``b``.

    IoU = |A ∩ B| / (|A| + |B| − |A ∩ B|), where the intersection is itself a box
    ``[max(x1), max(y1), min(x2), min(y2)]`` clamped to non-negative width/height.

    Args:
        a: (N, 4) xyxy.  b: (M, 4) xyxy.
    Returns:
        (N, M) IoU matrix.
    """
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # (N, M, 2) top-left of intersection
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2) bottom-right of intersection
    wh = np.clip(rb - lt, 0.0, None)  # (N, M, 2) clamp: disjoint boxes give 0
    inter = wh[..., 0] * wh[..., 1]  # (N, M)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter  # (N, M)
    return inter / np.maximum(union, 1e-12)  # (N, M)


def _enclosing_box(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest box containing each pair. (N, 4), (M, 4) → (N, M, 4)."""
    lt = np.minimum(a[:, None, :2], b[None, :, :2])  # (N, M, 2)
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2)
    return np.concatenate([lt, rb], axis=-1)  # (N, M, 4)


def box_giou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generalised IoU: ``IoU − |C \\ (A ∪ B)| / |C|`` with ``C`` the enclosing box. (N, M), in (−1, 1]."""
    iou = box_iou(a, b)  # (N, M)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # (N, M, 2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2)
    wh = np.clip(rb - lt, 0.0, None)  # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter  # (N, M)
    enc = _enclosing_box(a, b)  # (N, M, 4)
    enc_area = (enc[..., 2] - enc[..., 0]) * (enc[..., 3] - enc[..., 1])  # (N, M)
    return iou - (enc_area - union) / np.maximum(enc_area, 1e-12)  # (N, M)


def box_diou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance IoU: ``IoU − ρ²(centres) / c²`` with ``c`` the enclosing-box diagonal. (N, M)."""
    iou = box_iou(a, b)  # (N, M)
    ca = xyxy_to_cxcywh(a)[:, :2]  # (N, 2) centres
    cb = xyxy_to_cxcywh(b)[:, :2]  # (M, 2)
    rho2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)  # (N, M) squared centre distance
    enc = _enclosing_box(a, b)  # (N, M, 4)
    c2 = (enc[..., 2] - enc[..., 0]) ** 2 + (enc[..., 3] - enc[..., 1]) ** 2  # (N, M) diagonal²
    return iou - rho2 / np.maximum(c2, 1e-12)  # (N, M)


def box_ciou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Complete IoU: DIoU minus an aspect-ratio term ``α v``, ``v = (4/π²)(atan(w_b/h_b) − atan(w_a/h_a))²``."""
    iou = box_iou(a, b)  # (N, M)
    diou = box_diou(a, b)  # (N, M)
    wa, ha = (a[:, 2] - a[:, 0])[:, None], (a[:, 3] - a[:, 1])[:, None]  # (N, 1)
    wb, hb = (b[:, 2] - b[:, 0])[None, :], (b[:, 3] - b[:, 1])[None, :]  # (1, M)
    v = (4.0 / np.pi**2) * (np.arctan(wb / hb) - np.arctan(wa / ha)) ** 2  # (N, M)
    alpha = v / np.maximum(1.0 - iou + v, 1e-12)  # (N, M) trade-off weight
    return diou - alpha * v  # (N, M)


# ---------------------------------------------------------------------------
# Δ-encoding (Faster R-CNN / SSD): regress offsets relative to an anchor
# ---------------------------------------------------------------------------


def encode_boxes(gt: np.ndarray, anchors: np.ndarray, variances: tuple[float, float] = (0.1, 0.2)) -> np.ndarray:
    """Targets ``t = ((g_cx − a_cx)/(a_w σ₁), (g_cy − a_cy)/(a_h σ₁), log(g_w/a_w)/σ₂, log(g_h/a_h)/σ₂)``.

    Dividing by the "variances" (SSD naming; ``1/weights`` in torchvision) rescales the
    targets to roughly unit variance so one smooth-L1 works for centre and size.

    Args:
        gt: (N, 4) xyxy matched ground truth.  anchors: (N, 4) xyxy.
    Returns:
        (N, 4) regression targets.
    """
    g = xyxy_to_cxcywh(gt)  # (N, 4)
    a = xyxy_to_cxcywh(anchors)  # (N, 4)
    t_xy = (g[:, :2] - a[:, :2]) / (a[:, 2:] * variances[0])  # (N, 2)
    t_wh = np.log(g[:, 2:] / a[:, 2:]) / variances[1]  # (N, 2)
    return np.concatenate([t_xy, t_wh], axis=1)  # (N, 4)


def decode_boxes(deltas: np.ndarray, anchors: np.ndarray, variances: tuple[float, float] = (0.1, 0.2), max_wh_log: float = 4.135) -> np.ndarray:
    """Inverse of :func:`encode_boxes`; ``exp`` is clamped (``log(1000/16)``) to stop blow-ups early in training.

    Args:
        deltas: (N, 4).  anchors: (N, 4) xyxy.
    Returns:
        (N, 4) xyxy boxes.
    """
    a = xyxy_to_cxcywh(anchors)  # (N, 4)
    cxcy = a[:, :2] + deltas[:, :2] * variances[0] * a[:, 2:]  # (N, 2)
    wh = a[:, 2:] * np.exp(np.minimum(deltas[:, 2:] * variances[1], max_wh_log))  # (N, 2)
    return cxcywh_to_xyxy(np.concatenate([cxcy, wh], axis=1))  # (N, 4)
