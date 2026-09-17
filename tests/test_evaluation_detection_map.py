import numpy as np

from mlbook.evaluation import detection_map as dm

B = lambda *a: np.array(a, dtype=float)  # noqa: E731


def test_iou_matrix_hand_computed():
    A = np.array([[0, 0, 2, 2]], dtype=float)
    Bm = np.array([[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6]], dtype=float)
    assert np.allclose(dm.iou_matrix(A, Bm), [[1 / 7, 1.0, 0.0]])


def _case():
    gts = [dm.GroundTruth(0, 0, B(0, 0, 10, 10)), dm.GroundTruth(0, 0, B(20, 20, 30, 30)),
           dm.GroundTruth(1, 0, B(0, 0, 10, 10))]
    dets = [
        dm.Detection(0, 0, 0.9, B(0, 0, 10, 10)),      # TP
        dm.Detection(0, 0, 0.8, B(1, 1, 10, 10)),      # duplicate -> FP
        dm.Detection(1, 0, 0.7, B(50, 50, 60, 60)),    # FP
        dm.Detection(0, 0, 0.6, B(20, 20, 30, 30)),    # TP
    ]
    return dets, gts


def test_match_detections_greedy_and_duplicates():
    dets, gts = _case()
    tp, fp, n_gt = dm.match_detections(dets, gts, 0, 0.5)
    assert tp.tolist() == [1, 0, 0, 1] and fp.tolist() == [0, 1, 1, 0] and n_gt == 3


def test_precision_recall_from_matches():
    p, r = dm.precision_recall_from_matches(np.array([1, 0, 0, 1.0]), np.array([0, 1, 1, 0.0]), 3)
    assert np.allclose(p, [1, 1 / 2, 1 / 3, 1 / 2]) and np.allclose(r, [1 / 3, 1 / 3, 1 / 3, 2 / 3])


def test_ap_interpolations_hand_computed():
    p = np.array([1, 1 / 2, 1 / 3, 1 / 2])
    r = np.array([1 / 3, 1 / 3, 1 / 3, 2 / 3])
    # all-points: envelope = 1 on r<=1/3, 1/2 on (1/3, 2/3], 0 after
    assert np.isclose(dm.ap_all_points(p, r), 1 / 3 * 1 + 1 / 3 * 1 / 2)
    # voc07: r in {0,.1,.2,.3} -> 1 ; {.4,.5,.6} -> 1/2 ; {.7,...,1} -> 0
    assert np.isclose(dm.ap_voc07(p, r), (4 * 1 + 3 * 0.5) / 11)
    c = dm.ap_coco101(p, r)
    assert 0.45 < c < 0.55


def test_average_precision_and_map_end_to_end():
    dets, gts = _case()
    assert np.isclose(dm.average_precision(dets, gts, 0, 0.5, "all_point"), 0.5)
    assert np.isnan(dm.average_precision(dets, gts, 5, 0.5))  # class without GT
    m = dm.mean_average_precision(dets, gts, classes=[0, 5], iou_thrs=(0.5, 0.95))
    assert np.isclose(m, 0.5)  # exact boxes -> same AP at every IoU
    assert len(dm.COCO_IOU_THRESHOLDS) == 10
