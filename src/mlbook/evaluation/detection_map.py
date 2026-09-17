"""Object-detection AP / mAP from scratch (NumPy).

Boxes are (x1, y1, x2, y2). For one class: sort detections by score, greedily
match each to the best unmatched ground truth of the same image with IoU >= thr
(a second match to the same GT is a false positive), accumulate TP/FP, build the
precision/recall curve, then integrate it with one of three conventions:
    voc07     11-point: mean over r in {0, 0.1, ..., 1} of max precision at recall >= r
    all_point VOC2010+ / "area": make precision monotone from the right, sum rectangles
    coco101   101-point: like voc07 with r in {0, 0.01, ..., 1}
COCO mAP@[.5:.95] averages the all-points-style 101-point AP over IoU thresholds
0.50, 0.55, ..., 0.95 and over classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    image_id: int
    cls: int
    score: float
    box: np.ndarray  # (4,)


@dataclass
class GroundTruth:
    image_id: int
    cls: int
    box: np.ndarray  # (4,)


def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise IoU. A (n, 4), B (m, 4) -> (n, m)."""
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])  # (n, m)
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])  # (n, m)
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])  # (n, m)
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])  # (n, m)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)  # (n, m)
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])  # (n,)
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])  # (m,)
    union = area_a[:, None] + area_b[None, :] - inter  # (n, m)
    return inter / np.maximum(union, 1e-12)  # (n, m)


def match_detections(
    dets: list[Detection], gts: list[GroundTruth], cls: int, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy matching for one class. Returns (tp (D,), fp (D,), n_gt) in score order."""
    d = sorted([x for x in dets if x.cls == cls], key=lambda x: -x.score)
    g = [x for x in gts if x.cls == cls]
    gt_by_img: dict[int, list[int]] = {}
    for j, x in enumerate(g):
        gt_by_img.setdefault(x.image_id, []).append(j)
    used = np.zeros(len(g), dtype=bool)  # (G,)
    tp = np.zeros(len(d))  # (D,)
    fp = np.zeros(len(d))  # (D,)
    for i, det in enumerate(d):
        cand = gt_by_img.get(det.image_id, [])
        if not cand:
            fp[i] = 1
            continue
        G = np.stack([g[j].box for j in cand])  # (n_img_gt, 4)
        ious = iou_matrix(det.box[None, :], G)[0]  # (n_img_gt,)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not used[cand[best]]:
            tp[i] = 1
            used[cand[best]] = True
        else:
            fp[i] = 1
    return tp, fp, len(g)


def precision_recall_from_matches(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative precision/recall along the score-sorted detections. -> (D,), (D,)."""
    ctp = np.cumsum(tp)  # (D,)
    cfp = np.cumsum(fp)  # (D,)
    recall = ctp / max(n_gt, 1)  # (D,)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)  # (D,)
    return precision, recall


def ap_voc07(precision: np.ndarray, recall: np.ndarray) -> float:
    """11-point interpolated AP (VOC 2007)."""
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = precision[recall >= r]
        ap += (p.max() if p.size else 0.0) / 11.0
    return float(ap)


def ap_all_points(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the monotone (right-to-left max) precision envelope (VOC 2010+)."""
    mrec = np.r_[0.0, recall, 1.0]  # (D+2,)
    mpre = np.r_[0.0, precision, 0.0]  # (D+2,)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])  # make precision non-increasing
    idx = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1  # where recall changes
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def ap_coco101(precision: np.ndarray, recall: np.ndarray) -> float:
    """101-point interpolation on the monotone envelope (COCO convention)."""
    mpre = precision.copy()
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += (mpre[idx] if idx < len(mpre) else 0.0) / 101.0
    return float(ap)


def average_precision(
    dets: list[Detection], gts: list[GroundTruth], cls: int, iou_thr: float = 0.5, method: str = "all_point"
) -> float:
    tp, fp, n_gt = match_detections(dets, gts, cls, iou_thr)
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    precision, recall = precision_recall_from_matches(tp, fp, n_gt)
    if method == "voc07":
        return ap_voc07(precision, recall)
    if method == "coco101":
        return ap_coco101(precision, recall)
    return ap_all_points(precision, recall)


def mean_average_precision(
    dets: list[Detection], gts: list[GroundTruth], classes: list[int],
    iou_thrs: tuple[float, ...] = (0.5,), method: str = "all_point",
) -> float:
    """Mean over classes (with any GT) and IoU thresholds. COCO: iou_thrs=0.5..0.95."""
    aps = []
    for thr in iou_thrs:
        for c in classes:
            ap = average_precision(dets, gts, c, thr, method)
            if not np.isnan(ap):
                aps.append(ap)
    return float(np.mean(aps)) if aps else float("nan")


COCO_IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
