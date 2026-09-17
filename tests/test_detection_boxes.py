import numpy as np

from mlbook.detection import boxes as B


def test_box_iou_hand_computed_cases():
    a = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 4.0, 4.0]])
    b = np.array([[5.0, 5.0, 15.0, 15.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    iou = B.box_iou(a, b)  # (2, 3)
    assert iou.shape == (2, 3)
    assert abs(iou[0, 0] - 25 / 175) < 1e-12  # inter 5x5=25, union 100+100-25
    assert iou[0, 1] == 1.0
    assert iou[0, 2] == 0.0  # disjoint → clamped to 0, never negative
    assert abs(iou[1, 1] - 16 / 100) < 1e-12  # containment: 16/100


def test_box_giou_diou_ciou_properties():
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array([[4.0, 0.0, 6.0, 2.0]])  # disjoint, IoU = 0 but GIoU < 0 (gradient signal exists)
    assert B.box_iou(a, b)[0, 0] == 0.0
    assert abs(B.box_giou(a, b)[0, 0] - (0 - (12 - 8) / 12)) < 1e-12  # enclosing 6x2=12, union 8
    assert abs(B.box_diou(a, b)[0, 0] - (0 - 16 / (36 + 4))) < 1e-12  # centre dist² 16, diag² 40
    same = np.array([[1.0, 1.0, 3.0, 5.0]])
    assert abs(B.box_ciou(same, same)[0, 0] - 1.0) < 1e-12  # identical boxes: all variants = 1
    farther = np.array([[8.0, 0.0, 10.0, 2.0]])
    assert B.box_diou(a, farther)[0, 0] < B.box_diou(a, b)[0, 0]  # DIoU penalises distance


def test_encode_decode_boxes_roundtrip_and_identity():
    anchors = np.array([[10.0, 10.0, 30.0, 50.0], [0.0, 0.0, 8.0, 8.0]])
    gt = np.array([[12.0, 8.0, 34.0, 52.0], [1.0, 1.0, 9.0, 11.0]])
    t = B.encode_boxes(gt, anchors)
    assert np.allclose(B.decode_boxes(t, anchors), gt)
    assert np.allclose(B.encode_boxes(anchors, anchors), 0.0)  # an anchor on its target has zero deltas
    # hand check row 0: a=(20,30,20,40), g=(23,30,22,44): tx=(3/20)/0.1=1.5, ty=0, tw=log(1.1)/0.2, th=log(1.1)/0.2
    assert np.allclose(t[0], [1.5, 0.0, np.log(1.1) / 0.2, np.log(1.1) / 0.2])


def test_xyxy_cxcywh_conversions():
    xyxy = np.array([[2.0, 4.0, 6.0, 10.0]])
    assert np.allclose(B.xyxy_to_cxcywh(xyxy), [[4.0, 7.0, 4.0, 6.0]])
    assert np.allclose(B.cxcywh_to_xyxy(B.xyxy_to_cxcywh(xyxy)), xyxy)
