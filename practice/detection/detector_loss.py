# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/detection/detector_loss.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k detector_loss -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py detection/detector_loss --force

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

def build_targets(anchors: np.ndarray, gt_boxes: np.ndarray, gt_labels: np.ndarray, num_classes: int, pos_iou: float=0.5, neg_iou: float=0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one image's GT into per-anchor classification and regression targets.

    Args:
        anchors: (N, 4).  gt_boxes: (M, 4).  gt_labels: (M,) ints in [0, K).
    Returns:
        cls_targets (N, K) one-hot (all-zero for negatives),
        reg_targets (N, 4) Δ-encoded (zeros for non-positives),
        labels (N,) ∈ {1, 0, −1}.
    """
    raise NotImplementedError('TODO: implement build_targets (see the reference in src/mlbook)')

def single_stage_loss(cls_logits: torch.Tensor, box_deltas: torch.Tensor, cls_targets: torch.Tensor, reg_targets: torch.Tensor, labels: torch.Tensor, alpha: float=0.25, gamma: float=2.0, beta: float=1.0 / 9.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Classification (focal, over non-ignored anchors) + regression (smooth-L1, positives only).

    Args:
        cls_logits: (N, K).  box_deltas: (N, 4).
        cls_targets: (N, K).  reg_targets: (N, 4).  labels: (N,) ∈ {1, 0, −1}.
    Returns:
        (cls_loss, reg_loss), each a scalar normalised by ``max(N_pos, 1)``.
    """
    raise NotImplementedError('TODO: implement single_stage_loss (see the reference in src/mlbook)')
