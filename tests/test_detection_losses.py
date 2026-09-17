import numpy as np
import torch

torch.set_num_threads(1)

from mlbook.detection import focal_loss as FL
from mlbook.detection.anchors import multilevel_anchors
from mlbook.detection.detector_loss import build_targets, single_stage_loss


def test_sigmoid_focal_loss_reduces_to_bce_at_gamma_zero():
    z = torch.randn(10, 3)
    y = (torch.rand(10, 3) > 0.5).float()
    fl = FL.sigmoid_focal_loss(z, y, alpha=0.5, gamma=0.0, reduction="none")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
    assert torch.allclose(fl, 0.5 * bce)


def test_sigmoid_focal_loss_downweights_easy_examples():
    easy = FL.sigmoid_focal_loss(torch.tensor([6.0]), torch.tensor([1.0]), alpha=1.0, gamma=2.0)
    bce_easy = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([6.0]), torch.tensor([1.0]))
    assert easy < 1e-3 * bce_easy  # (1 − p_t)^2 with p_t ≈ 0.9975 → factor ≈ 6e-6
    hard = FL.sigmoid_focal_loss(torch.tensor([-3.0]), torch.tensor([1.0]), alpha=1.0, gamma=2.0)
    bce_hard = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([-3.0]), torch.tensor([1.0]))
    assert hard > 0.8 * bce_hard  # hard examples keep most of their loss


def test_focal_loss_grad_numpy_matches_autograd():
    z = np.random.randn(50) * 3
    y = (np.random.rand(50) > 0.5).astype(np.int64)
    zt = torch.tensor(z, requires_grad=True)
    FL.sigmoid_focal_loss(zt, torch.tensor(y, dtype=torch.float64), alpha=0.25, gamma=2.0).backward()
    assert np.allclose(FL.focal_loss_grad_numpy(z, y), zt.grad.numpy(), atol=1e-10)


def test_prior_bias_init():
    b = FL.prior_bias_init(0.01)
    assert abs(torch.sigmoid(torch.tensor(b, dtype=torch.float64)).item() - 0.01) < 1e-9


def test_build_targets_and_single_stage_loss():
    anchors = multilevel_anchors((64, 64), strides=(8, 16, 32))
    gt_boxes = np.array([[8.0, 8.0, 40.0, 40.0], [30.0, 10.0, 62.0, 60.0]])
    gt_labels = np.array([2, 0])
    cls_t, reg_t, labels = build_targets(anchors, gt_boxes, gt_labels, num_classes=3)
    pos = labels == 1
    assert pos.sum() >= 2 and cls_t[pos].sum(1).min() == 1.0 and cls_t[~pos].sum() == 0.0
    N = len(anchors)
    logits = torch.zeros(N, 3) + FL.prior_bias_init(0.01)
    deltas = torch.tensor(reg_t)  # perfect regression → zero box loss
    cls_loss, reg_loss = single_stage_loss(logits, deltas, torch.tensor(cls_t), torch.tensor(reg_t), torch.tensor(labels))
    assert reg_loss.item() == 0.0
    # with the prior init the negatives' total focal loss is small relative to the positives'
    per_pos = cls_loss.item()
    assert 0.5 < per_pos < 10.0
    # gradient flows
    logits.requires_grad_(True)
    single_stage_loss(logits, deltas, torch.tensor(cls_t), torch.tensor(reg_t), torch.tensor(labels))[0].backward()
    assert logits.grad.abs().sum() > 0
