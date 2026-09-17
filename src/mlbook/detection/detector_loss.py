"""A simplified single-stage (RetinaNet-style) detection loss in PyTorch.

Per image: assign anchors to GT (NumPy assigner), then
    L = [ Σ_anchors FL(cls_logits, one-hot targets) + Σ_positives smoothL1(box_deltas, encoded targets) ] / N_pos.
Both terms are normalised by the number of positive anchors, not by the number of anchors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .anchors import assign_max_iou
from .boxes import encode_boxes
from .focal_loss import sigmoid_focal_loss


def build_targets(anchors: np.ndarray, gt_boxes: np.ndarray, gt_labels: np.ndarray, num_classes: int, pos_iou: float = 0.5, neg_iou: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one image's GT into per-anchor classification and regression targets.

    Args:
        anchors: (N, 4).  gt_boxes: (M, 4).  gt_labels: (M,) ints in [0, K).
    Returns:
        cls_targets (N, K) one-hot (all-zero for negatives),
        reg_targets (N, 4) Δ-encoded (zeros for non-positives),
        labels (N,) ∈ {1, 0, −1}.
    """
    N = len(anchors)
    labels, matched = assign_max_iou(anchors, gt_boxes, pos_iou, neg_iou)  # (N,), (N,)
    cls_targets = np.zeros((N, num_classes), dtype=np.float32)  # (N, K)
    reg_targets = np.zeros((N, 4), dtype=np.float32)  # (N, 4)
    pos = np.flatnonzero(labels == 1)  # (P,)
    if pos.size > 0:
        cls_targets[pos, gt_labels[matched[pos]]] = 1.0
        reg_targets[pos] = encode_boxes(gt_boxes[matched[pos]], anchors[pos])  # (P, 4)
    return cls_targets, reg_targets, labels


def single_stage_loss(cls_logits: torch.Tensor, box_deltas: torch.Tensor, cls_targets: torch.Tensor, reg_targets: torch.Tensor, labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, beta: float = 1.0 / 9.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Classification (focal, over non-ignored anchors) + regression (smooth-L1, positives only).

    Args:
        cls_logits: (N, K).  box_deltas: (N, 4).
        cls_targets: (N, K).  reg_targets: (N, 4).  labels: (N,) ∈ {1, 0, −1}.
    Returns:
        (cls_loss, reg_loss), each a scalar normalised by ``max(N_pos, 1)``.
    """
    valid = labels >= 0  # (N,) bool: drop ignored anchors from the classification term
    num_pos = (labels == 1).sum().clamp(min=1).to(cls_logits.dtype)  # scalar
    cls_loss = sigmoid_focal_loss(cls_logits[valid], cls_targets[valid], alpha, gamma, reduction="sum") / num_pos  # scalar
    pos = labels == 1  # (N,) bool
    reg_loss = F.smooth_l1_loss(box_deltas[pos], reg_targets[pos], beta=beta, reduction="sum") / num_pos  # scalar
    return cls_loss, reg_loss
