# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/evaluation/detection_map.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k detection_map -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py evaluation/detection_map --force

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
    box: np.ndarray

@dataclass
class GroundTruth:
    image_id: int
    cls: int
    box: np.ndarray

def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise IoU. A (n, 4), B (m, 4) -> (n, m)."""
    raise NotImplementedError('TODO: implement iou_matrix (see the reference in src/mlbook)')

def match_detections(dets: list[Detection], gts: list[GroundTruth], cls: int, iou_thr: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy matching for one class. Returns (tp (D,), fp (D,), n_gt) in score order."""
    raise NotImplementedError('TODO: implement match_detections (see the reference in src/mlbook)')

def precision_recall_from_matches(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative precision/recall along the score-sorted detections. -> (D,), (D,)."""
    raise NotImplementedError('TODO: implement precision_recall_from_matches (see the reference in src/mlbook)')

def ap_voc07(precision: np.ndarray, recall: np.ndarray) -> float:
    """11-point interpolated AP (VOC 2007)."""
    raise NotImplementedError('TODO: implement ap_voc07 (see the reference in src/mlbook)')

def ap_all_points(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the monotone (right-to-left max) precision envelope (VOC 2010+)."""
    raise NotImplementedError('TODO: implement ap_all_points (see the reference in src/mlbook)')

def ap_coco101(precision: np.ndarray, recall: np.ndarray) -> float:
    """101-point interpolation on the monotone envelope (COCO convention)."""
    raise NotImplementedError('TODO: implement ap_coco101 (see the reference in src/mlbook)')

def average_precision(dets: list[Detection], gts: list[GroundTruth], cls: int, iou_thr: float=0.5, method: str='all_point') -> float:
    raise NotImplementedError('TODO: implement average_precision (see the reference in src/mlbook)')

def mean_average_precision(dets: list[Detection], gts: list[GroundTruth], classes: list[int], iou_thrs: tuple[float, ...]=(0.5,), method: str='all_point') -> float:
    """Mean over classes (with any GT) and IoU thresholds. COCO: iou_thrs=0.5..0.95."""
    raise NotImplementedError('TODO: implement mean_average_precision (see the reference in src/mlbook)')
COCO_IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
