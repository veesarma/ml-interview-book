import numpy as np

from mlbook.detection import nms as N


def _random_boxes(n, rng):
    xy = rng.uniform(0, 80, size=(n, 2))
    wh = rng.uniform(5, 40, size=(n, 2))
    return np.concatenate([xy, xy + wh], axis=1)


def test_nms_matches_bruteforce_reference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        boxes = _random_boxes(60, rng)
        scores = rng.random(60)
        for thr in (0.3, 0.5, 0.7):
            assert np.array_equal(N.nms(boxes, scores, thr), N.nms_bruteforce(boxes, scores, thr))


def test_nms_hand_case():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 5, 5]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    keep = N.nms(boxes, scores, 0.5)
    # box1 overlaps box0 with IoU 81/119 ≈ 0.68 → suppressed; box3 has IoU 25/100 → kept
    assert keep.tolist() == [0, 2, 3]


def test_batched_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 1, 0])
    keep = N.batched_nms(boxes, scores, classes, 0.5)
    assert sorted(keep.tolist()) == [0, 1]  # same box, different class: both survive; box 2 (class 0) dies


def test_soft_nms_decays_instead_of_deleting():
    boxes = np.array([[0, 0, 10, 10], [2, 0, 12, 10], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.85, 0.5])
    keep, new_scores = N.soft_nms(boxes, scores, sigma=0.5, score_threshold=0.01)
    assert keep.tolist() == [0, 2, 1]  # box 1 survives with a decayed score, re-ranked below box 2
    iou = 8 * 10 / (100 + 100 - 80)
    assert abs(new_scores[2] - 0.85 * np.exp(-(iou**2) / 0.5)) < 1e-12
