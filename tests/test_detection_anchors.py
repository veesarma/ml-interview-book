import numpy as np

from mlbook.detection import anchors as A
from mlbook.detection.boxes import box_iou


def test_base_anchors_area_and_ratio():
    t = A.base_anchors(32.0, ratios=(0.5, 1.0, 2.0), scales=(1.0,))
    w = t[:, 2] - t[:, 0]
    h = t[:, 3] - t[:, 1]
    assert np.allclose(w * h, 32.0**2)  # constant area
    assert np.allclose(h / w, [0.5, 1.0, 2.0])  # requested aspect ratios
    assert np.allclose(t[:, :2] + t[:, 2:], 0.0)  # centred at the origin


def test_grid_anchors_centres_and_count():
    t = A.base_anchors(16.0, ratios=(1.0,), scales=(1.0,))
    anchors = A.grid_anchors(2, 3, stride=8, templates=t)
    assert anchors.shape == (6, 4)
    centres = (anchors[:, :2] + anchors[:, 2:]) / 2
    assert np.allclose(centres[0], [4.0, 4.0]) and np.allclose(centres[-1], [20.0, 12.0])  # (j+0.5)s, (i+0.5)s


def test_multilevel_anchors_count():
    anchors = A.multilevel_anchors((64, 64), strides=(8, 16, 32))
    assert anchors.shape == ((64 + 16 + 4) * 9, 4)


def test_assign_max_iou_thresholds_and_low_quality_match():
    anchors = np.array([[0, 0, 10, 10], [0, 0, 5, 5], [100, 100, 110, 110], [40, 40, 41, 41]], dtype=float)
    gt = np.array([[0, 0, 10, 10], [40, 40, 44, 44]], dtype=float)
    labels, matched = A.assign_max_iou(anchors, gt, pos_iou=0.5, neg_iou=0.4)
    assert labels[0] == 1 and matched[0] == 0  # IoU 1
    assert labels[1] == 0  # IoU 0.25 with gt0 < 0.4 → negative
    assert labels[2] == 0
    assert labels[3] == 1 and matched[3] == 1  # low-quality match: best anchor for gt1 (IoU 1/16) still positive


def test_assign_max_iou_negative_band():
    anchors = np.array([[0, 0, 5, 5]], dtype=float)
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    labels, _ = A.assign_max_iou(anchors, gt, pos_iou=0.5, neg_iou=0.4)
    assert labels[0] == 1  # only anchor → claimed by the GT regardless of its 0.25 IoU


def test_assign_atss_adaptive_threshold():
    anchors = A.multilevel_anchors((64, 64), strides=(8, 16))
    level_ids = np.concatenate([np.zeros(64 * 9, dtype=int), np.ones(16 * 9, dtype=int)])
    gt = np.array([[10.0, 10.0, 34.0, 30.0]])
    labels, matched = A.assign_atss(anchors, gt, level_ids, top_k=9)
    pos = np.flatnonzero(labels == 1)
    assert 1 <= len(pos) <= 18
    assert np.all(matched[pos] == 0)
    centres = (anchors[pos, :2] + anchors[pos, 2:]) / 2
    assert np.all((centres > gt[0, :2]) & (centres < gt[0, 2:]))  # positives lie inside the GT
    assert box_iou(anchors[pos], gt).min() > 0.0
