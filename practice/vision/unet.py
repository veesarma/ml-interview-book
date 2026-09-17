# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/vision/unet.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k unet -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py vision/unet --force

"""A tiny U-Net, segmentation losses (cross-entropy, Dice, boundary) and metrics (mIoU, PQ).

Shapes: images ``(B, C, H, W)``, logits ``(B, K, H, W)``, integer masks ``(B, H, W)``.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class DoubleConv(nn.Module):
    """Two ``conv3x3 → BN → ReLU`` units: the U-Net building block. (B,C_in,H,W) → (B,C_out,H,W)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyUNet(nn.Module):
    """Three-level U-Net: encoder (down ×2), bottleneck, decoder (up ×2) with skip concatenation.

    Encoder:  x (B,C,H,W) → e1 (B,w,H,W) → e2 (B,2w,H/2,W/2) → e3 (B,4w,H/4,W/4)
    Decoder:  up(e3) ‖ e2 → d2 (B,2w,H/2,W/2);  up(d2) ‖ e1 → d1 (B,w,H,W) → logits (B,K,H,W)
    The skips carry full-resolution edges that pooling destroyed, so boundaries are sharp.
    """

    def __init__(self, in_channels: int=1, num_classes: int=3, width: int=16) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float=1.0) -> torch.Tensor:
    """Soft Dice loss averaged over classes: ``1 − (2 Σ p·y + ε) / (Σ p + Σ y + ε)``.

    Dice is a region-overlap objective (like F1 on pixels), so it is insensitive to
    class imbalance in a way per-pixel cross-entropy is not.

    Args:
        logits: (B, K, H, W).  target: (B, H, W) integer labels in [0, K).
    Returns:
        scalar.
    """
    raise NotImplementedError('TODO: implement dice_loss (see the reference in src/mlbook)')

def boundary_weight_map(target: torch.Tensor, width: int=1, boundary_weight: float=5.0) -> torch.Tensor:
    """Per-pixel weights that up-weight pixels within ``width`` of a label boundary.

    A boundary pixel is one whose label differs from a neighbour; the map is dilated
    ``width`` times with a 3×3 max-pool.  Multiplying cross-entropy by this map is the
    simplest "boundary loss" (the original U-Net used a morphology-based weight map).

    Args:
        target: (B, H, W) integer labels.
    Returns:
        (B, H, W) weights in {1, boundary_weight}.
    """
    raise NotImplementedError('TODO: implement boundary_weight_map (see the reference in src/mlbook)')

def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float=1.0, boundary_weight: float=0.0) -> torch.Tensor:
    """``CE (optionally boundary-weighted) + dice_weight · Dice``.  Returns a scalar."""
    raise NotImplementedError('TODO: implement segmentation_loss (see the reference in src/mlbook)')

def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``M[t, p]`` = number of pixels with true class ``t`` predicted as ``p``. Inputs (…,) ints."""
    raise NotImplementedError('TODO: implement confusion_matrix (see the reference in src/mlbook)')

def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Dataset-level mIoU: ``mean_k  TP_k / (TP_k + FP_k + FN_k)`` over classes present in GT or pred."""
    raise NotImplementedError('TODO: implement mean_iou (see the reference in src/mlbook)')

def panoptic_quality(pred_segments: list[tuple[int, torch.Tensor]], gt_segments: list[tuple[int, torch.Tensor]], iou_threshold: float=0.5) -> tuple[float, float, float]:
    """PQ = SQ × RQ on one image, single-class version for clarity.

    Each segment is ``(class_id, bool mask (H, W))``.  A prediction matches a GT segment iff
    same class and IoU > 0.5 (this makes matching unique).
    Returns:
        (PQ, SQ = mean matched IoU, RQ = TP / (TP + FP/2 + FN/2)).
    """
    raise NotImplementedError('TODO: implement panoptic_quality (see the reference in src/mlbook)')

def synthetic_segmentation(n: int, size: int=32, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Images with a disc (class 1) and a rectangle (class 2) on background (class 0).

    Returns:
        images (n, 1, S, S), masks (n, S, S) int64.
    """
    raise NotImplementedError('TODO: implement synthetic_segmentation (see the reference in src/mlbook)')
