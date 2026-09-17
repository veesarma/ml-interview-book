import numpy as np
import torch

from mlbook.perception import camera_rig as cr
from mlbook.perception import fusion as fu


def test_box_iou_known_values():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])
    assert np.allclose(fu.box_iou(a, b), [[25.0 / 175.0, 0.0]])


def test_weighted_boxes_fusion_merges_agreeing_sensors():
    cam = np.array([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 120.0, 120.0]])
    lidar = np.array([[12.0, 8.0, 52.0, 48.0]])
    boxes, scores = fu.weighted_boxes_fusion([cam, lidar], [np.array([0.9, 0.8]), np.array([0.7])], iou_thr=0.55)
    assert boxes.shape == (2, 4)
    expected = (0.9 * cam[0] + 0.7 * lidar[0]) / 1.6
    assert np.allclose(boxes[0], expected)
    assert np.isclose(scores[0], 0.8)  # mean(0.9, 0.7) · min(2, 2)/2
    assert np.isclose(scores[1], 0.4)  # seen by one of two sensors: 0.8 · 1/2


def test_fuse_detection_probabilities_log_odds():
    p = fu.fuse_detection_probabilities(np.array([[0.7, 0.5, 0.9], [0.7, 0.5, 0.1]]))
    assert np.isclose(p[0], 0.49 / (0.49 + 0.09))  # 0.845: two agreeing 0.7s
    assert np.isclose(p[1], 0.5)  # uninformative sensors leave the prior
    assert np.isclose(p[2], 0.5)  # 0.9 and 0.1 cancel exactly


def test_point_painting_appends_pixel_scores():
    cam = cr.make_surround_rig(n_cams=1, image_hw=(64, 64))[0]
    seg = np.zeros((3, 64, 64))
    seg[1, :, :32] = 1.0  # class 1 on the left half of the image
    seg[2, :, 32:] = 1.0  # class 2 on the right half
    pts = np.array([[11.0, 2.0, 1.6], [11.0, -2.0, 1.6], [-5.0, 0.0, 1.6]])  # left, right, behind camera
    painted = fu.point_painting(pts, seg, cam)
    assert painted.shape == (3, 6)
    assert np.allclose(painted[0, 3:], [0.0, 1.0, 0.0])
    assert np.allclose(painted[1, 3:], [0.0, 0.0, 1.0])
    assert np.allclose(painted[2, 3:], 0.0)


def test_radar_points_to_bev_channels():
    pts = np.array([[1.5, 1.5, 4.0, 10.0], [1.5, 1.5, 6.0, 20.0], [50.0, 0.0, 1.0, 1.0]])
    bev = fu.radar_points_to_bev(pts, (0.0, 4.0, 0.0, 4.0), (4, 4))
    assert bev.shape == (3, 4, 4)
    assert bev[0, 1, 1] == 2 and np.isclose(bev[1, 1, 1], 5.0) and bev[2, 1, 1] == 20.0
    assert bev[0].sum() == 2


def test_cross_attention_fusion_matches_torch_multihead_attention():
    torch.manual_seed(0)
    d, h = 16, 4
    ours = fu.CrossAttentionFusion(d, h)
    ref = torch.nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight], 0))
        ref.in_proj_bias.copy_(torch.cat([ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias], 0))
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        ref.out_proj.bias.copy_(ours.out_proj.bias)
    lidar, cam = torch.randn(2, 5, d), torch.randn(2, 7, d)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[1, 4:] = True
    out = ours(lidar, cam, mask)
    ref_out, _ = ref(lidar, cam, cam, key_padding_mask=mask)
    assert torch.allclose(out, lidar + ref_out, atol=1e-5)


def test_cross_attention_fusion_degrades_to_lidar_only_when_camera_missing():
    torch.manual_seed(0)
    block = fu.CrossAttentionFusion(8, 2)
    lidar, cam = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    out = block(lidar, cam, torch.ones(1, 4, dtype=torch.bool))
    assert torch.allclose(out, lidar + block.out_proj.bias)  # zero context → only the bias remains
    assert torch.isfinite(out).all()
