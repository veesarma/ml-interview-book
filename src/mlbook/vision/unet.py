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
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)  # (B, C_out, H, W)


class TinyUNet(nn.Module):
    """Three-level U-Net: encoder (down ×2), bottleneck, decoder (up ×2) with skip concatenation.

    Encoder:  x (B,C,H,W) → e1 (B,w,H,W) → e2 (B,2w,H/2,W/2) → e3 (B,4w,H/4,W/4)
    Decoder:  up(e3) ‖ e2 → d2 (B,2w,H/2,W/2);  up(d2) ‖ e1 → d1 (B,w,H,W) → logits (B,K,H,W)
    The skips carry full-resolution edges that pooling destroyed, so boundaries are sharp.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, width: int = 16) -> None:
        super().__init__()
        w = width
        self.enc1 = DoubleConv(in_channels, w)
        self.enc2 = DoubleConv(w, 2 * w)
        self.enc3 = DoubleConv(2 * w, 4 * w)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, kernel_size=2, stride=2)  # k = s: no checkerboard overlap
        self.dec2 = DoubleConv(4 * w, 2 * w)  # input is concat(up, skip) = 2w + 2w
        self.up1 = nn.ConvTranspose2d(2 * w, w, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(2 * w, w)  # w + w
        self.out = nn.Conv2d(w, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # (B, w, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 2w, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 4w, H/4, W/4)  bottleneck
        d2 = self.up2(e3)  # (B, 2w, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 2w, H/2, W/2)  skip: concat along channels
        d1 = self.up1(d2)  # (B, w, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, w, H, W)
        return self.out(d1)  # (B, K, H, W) per-pixel logits


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss averaged over classes: ``1 − (2 Σ p·y + ε) / (Σ p + Σ y + ε)``.

    Dice is a region-overlap objective (like F1 on pixels), so it is insensitive to
    class imbalance in a way per-pixel cross-entropy is not.

    Args:
        logits: (B, K, H, W).  target: (B, H, W) integer labels in [0, K).
    Returns:
        scalar.
    """
    K = logits.shape[1]
    probs = torch.softmax(logits, dim=1)  # (B, K, H, W)
    onehot = F.one_hot(target, K).permute(0, 3, 1, 2).to(probs.dtype)  # (B, K, H, W)
    intersection = (probs * onehot).sum(dim=(0, 2, 3))  # (K,)
    cardinality = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))  # (K,)
    dice = (2.0 * intersection + eps) / (cardinality + eps)  # (K,)
    return 1.0 - dice.mean()  # scalar


def boundary_weight_map(target: torch.Tensor, width: int = 1, boundary_weight: float = 5.0) -> torch.Tensor:
    """Per-pixel weights that up-weight pixels within ``width`` of a label boundary.

    A boundary pixel is one whose label differs from a neighbour; the map is dilated
    ``width`` times with a 3×3 max-pool.  Multiplying cross-entropy by this map is the
    simplest "boundary loss" (the original U-Net used a morphology-based weight map).

    Args:
        target: (B, H, W) integer labels.
    Returns:
        (B, H, W) weights in {1, boundary_weight}.
    """
    t = target.to(torch.float32)[:, None]  # (B, 1, H, W)
    dilated = F.max_pool2d(t, kernel_size=3, stride=1, padding=1)  # (B, 1, H, W)
    eroded = -F.max_pool2d(-t, kernel_size=3, stride=1, padding=1)  # (B, 1, H, W)
    boundary = (dilated != eroded).to(torch.float32)  # (B, 1, H, W)  label changes nearby
    for _ in range(width - 1):
        boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)  # widen the band
    weights = 1.0 + (boundary_weight - 1.0) * boundary  # (B, 1, H, W)
    return weights[:, 0]  # (B, H, W)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0, boundary_weight: float = 0.0) -> torch.Tensor:
    """``CE (optionally boundary-weighted) + dice_weight · Dice``.  Returns a scalar."""
    ce_map = F.cross_entropy(logits, target, reduction="none")  # (B, H, W)
    if boundary_weight > 0:
        weights = boundary_weight_map(target, boundary_weight=boundary_weight)  # (B, H, W)
        ce = (ce_map * weights).sum() / weights.sum()  # scalar
    else:
        ce = ce_map.mean()  # scalar
    return ce + dice_weight * dice_loss(logits, target)  # scalar


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``M[t, p]`` = number of pixels with true class ``t`` predicted as ``p``. Inputs (…,) ints."""
    idx = target.reshape(-1) * num_classes + pred.reshape(-1)  # (N,) combined index
    counts = torch.bincount(idx, minlength=num_classes * num_classes)  # (K·K,)
    return counts.reshape(num_classes, num_classes)  # (K, K)


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Dataset-level mIoU: ``mean_k  TP_k / (TP_k + FP_k + FN_k)`` over classes present in GT or pred."""
    M = confusion_matrix(pred, target, num_classes).to(torch.float64)  # (K, K)
    tp = M.diag()  # (K,)
    fp = M.sum(dim=0) - tp  # (K,) predicted k but not k
    fn = M.sum(dim=1) - tp  # (K,) is k but predicted otherwise
    denom = tp + fp + fn  # (K,)
    valid = denom > 0  # (K,)
    return (tp[valid] / denom[valid]).mean()  # scalar


def panoptic_quality(pred_segments: list[tuple[int, torch.Tensor]], gt_segments: list[tuple[int, torch.Tensor]], iou_threshold: float = 0.5) -> tuple[float, float, float]:
    """PQ = SQ × RQ on one image, single-class version for clarity.

    Each segment is ``(class_id, bool mask (H, W))``.  A prediction matches a GT segment iff
    same class and IoU > 0.5 (this makes matching unique).
    Returns:
        (PQ, SQ = mean matched IoU, RQ = TP / (TP + FP/2 + FN/2)).
    """
    matched_ious: list[float] = []
    used_pred: set[int] = set()
    for _, gt_mask in gt_segments:
        for p_idx, (p_cls, p_mask) in enumerate(pred_segments):
            if p_idx in used_pred:
                continue
            inter = (gt_mask & p_mask).sum().item()
            union = (gt_mask | p_mask).sum().item()
            iou = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                matched_ious.append(iou)
                used_pred.add(p_idx)
                break
    tp = len(matched_ious)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    if tp == 0:
        return 0.0, 0.0, 0.0
    sq = sum(matched_ious) / tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return sq * rq, sq, rq


def synthetic_segmentation(n: int, size: int = 32, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Images with a disc (class 1) and a rectangle (class 2) on background (class 0).

    Returns:
        images (n, 1, S, S), masks (n, S, S) int64.
    """
    g = generator if generator is not None else torch.Generator().manual_seed(0)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")  # (S, S) each
    images = 0.2 * torch.rand(n, 1, size, size, generator=g)  # (n, 1, S, S)
    masks = torch.zeros(n, size, size, dtype=torch.int64)  # (n, S, S)
    for i in range(n):
        cy, cx = torch.randint(8, size - 8, (2,), generator=g).tolist()
        r = int(torch.randint(3, 7, (1,), generator=g))
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r  # (S, S) bool
        t, l = torch.randint(2, size - 10, (2,), generator=g).tolist()
        h, w = torch.randint(4, 9, (2,), generator=g).tolist()
        rect = (yy >= t) & (yy < t + h) & (xx >= l) & (xx < l + w)  # (S, S) bool
        masks[i][rect] = 2
        masks[i][disc] = 1  # disc drawn last, on top
        images[i, 0][rect] = 0.6
        images[i, 0][disc] = 1.0
    return images, masks
