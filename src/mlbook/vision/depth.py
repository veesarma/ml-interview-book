"""Depth from stereo and monocular self-supervision: disparity ↔ depth, block matching,
cost volumes, and the photometric (SSIM + L1) reconstruction loss.

Stereo geometry (rectified pair, baseline ``b`` metres, focal ``f`` pixels):

    disparity d = x_L − x_R = f · b / Z      ⇔      Z = f · b / d.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def disparity_to_depth(disparity: np.ndarray, focal_px: float, baseline_m: float, eps: float = 1e-6) -> np.ndarray:
    """``Z = f·b / d``; zero disparity maps to ``inf`` (point at infinity).  Any shape."""
    return focal_px * baseline_m / np.maximum(disparity, eps)


def depth_to_disparity(depth: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    """``d = f·b / Z``.  Any shape."""
    return focal_px * baseline_m / depth


def depth_error_from_disparity_error(depth: np.ndarray, focal_px: float, baseline_m: float, disp_err_px: float) -> np.ndarray:
    """First-order depth error ``|ΔZ| ≈ Z² · Δd / (f·b)`` — error grows *quadratically* with range."""
    return depth**2 * disp_err_px / (focal_px * baseline_m)


def stereo_block_matching(left: np.ndarray, right: np.ndarray, max_disparity: int, block: int = 5) -> np.ndarray:
    """Winner-take-all SAD block matching on a rectified grayscale pair.

    For each candidate ``d`` build the cost ``C[d, i, j] = Σ_window |L(i, j) − R(i, j − d)|``
    (a 3-D *cost volume*), then take ``argmin_d``.  Left border columns where ``j − d < 0``
    get maximal cost.

    Args:
        left, right: (H, W) float images.
    Returns:
        (H, W) integer disparities in [0, max_disparity).
    """
    H, W = left.shape
    r = block // 2
    box = np.ones(block) / block  # (block,) box filter for window sums
    cost = np.full((max_disparity, H, W), np.inf)  # (D, H, W) cost volume
    for d in range(max_disparity):
        shifted = np.full_like(right, np.nan)  # (H, W)
        shifted[:, d:] = right[:, : W - d]  # R(i, j − d)
        diff = np.abs(left - shifted)  # (H, W) NaN where undefined
        diff = np.where(np.isnan(diff), 1e3, diff)  # heavy penalty off-image
        padded = np.pad(diff, r, mode="edge")  # (H+2r, W+2r)
        agg = np.apply_along_axis(np.convolve, 1, padded, box, mode="valid")  # (H+2r, W)
        agg = np.apply_along_axis(np.convolve, 0, agg, box, mode="valid")  # (H, W) window SAD
        cost[d] = agg
    return np.argmin(cost, axis=0)  # (H, W)


# ---------------------------------------------------------------------------
# Differentiable pieces used by learned stereo / self-supervised monodepth
# ---------------------------------------------------------------------------


def build_cost_volume(feat_left: torch.Tensor, feat_right: torch.Tensor, max_disparity: int) -> torch.Tensor:
    """Concatenation cost volume (PSMNet-style) for 3-D conv aggregation.

    Args:
        feat_left, feat_right: (B, C, H, W) feature maps.
    Returns:
        (B, 2C, D, H, W): at disparity ``d`` the right features are shifted right by ``d``.
    """
    B, C, H, W = feat_left.shape
    volume = feat_left.new_zeros(B, 2 * C, max_disparity, H, W)  # (B, 2C, D, H, W)
    for d in range(max_disparity):
        volume[:, :C, d, :, d:] = feat_left[:, :, :, d:]  # (B, C, H, W−d)
        volume[:, C:, d, :, d:] = feat_right[:, :, :, : W - d]  # (B, C, H, W−d)
    return volume


def soft_argmin_disparity(cost: torch.Tensor) -> torch.Tensor:
    """Differentiable disparity ``d̂ = Σ_d d · softmax(−cost)_d`` (GC-Net).  ``cost``: (B, D, H, W) → (B, H, W)."""
    D = cost.shape[1]
    probs = torch.softmax(-cost, dim=1)  # (B, D, H, W)
    disp_values = torch.arange(D, dtype=cost.dtype, device=cost.device)[None, :, None, None]  # (1, D, 1, 1)
    return (probs * disp_values).sum(dim=1)  # (B, H, W)


def warp_right_to_left(right: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    """Reconstruct the left image by sampling the right at ``x − d`` with bilinear ``grid_sample``.

    Args:
        right: (B, C, H, W).  disparity: (B, H, W) in pixels (positive = shift left).
    Returns:
        (B, C, H, W) synthesised left view.
    """
    B, _, H, W = right.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=right.dtype), torch.arange(W, dtype=right.dtype), indexing="ij")  # (H, W)
    x_src = xs[None] - disparity  # (B, H, W)  source column in the right image
    grid_x = 2.0 * x_src / (W - 1) - 1.0  # (B, H, W)  normalise to [−1, 1]
    grid_y = (2.0 * ys / (H - 1) - 1.0)[None].expand(B, H, W)  # (B, H, W)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)  grid_sample wants (x, y)
    return F.grid_sample(right, grid, mode="bilinear", padding_mode="border", align_corners=True)  # (B, C, H, W)


def ssim(x: torch.Tensor, y: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    """Per-pixel SSIM with a 3×3 average window (as in Monodepth).  Inputs (B, C, H, W) → (B, C, H, W)."""
    mu_x = F.avg_pool2d(x, 3, 1, 1)  # (B, C, H, W)
    mu_y = F.avg_pool2d(y, 3, 1, 1)  # (B, C, H, W)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x**2  # (B, C, H, W)
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y**2  # (B, C, H, W)
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y  # (B, C, H, W)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)  # (B, C, H, W)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)  # (B, C, H, W)
    return numerator / denominator  # (B, C, H, W)


def photometric_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    """Monodepth appearance loss ``α (1−SSIM)/2 + (1−α) |pred − target|`` averaged. Inputs (B, C, H, W)."""
    l1 = (pred - target).abs()  # (B, C, H, W)
    dssim = (1.0 - ssim(pred, target)) / 2.0  # (B, C, H, W)
    return (alpha * dssim + (1.0 - alpha) * l1).mean()  # scalar


def smoothness_loss(disparity: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware smoothness ``|∂x d| e^{−|∂x I|} + |∂y d| e^{−|∂y I|}``: smooth except at image edges.

    Args:
        disparity: (B, H, W) (use mean-normalised disparity).  image: (B, C, H, W).
    """
    dx_d = (disparity[:, :, 1:] - disparity[:, :, :-1]).abs()  # (B, H, W−1)
    dy_d = (disparity[:, 1:, :] - disparity[:, :-1, :]).abs()  # (B, H−1, W)
    dx_i = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1)  # (B, H, W−1)
    dy_i = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1)  # (B, H−1, W)
    return (dx_d * torch.exp(-dx_i)).mean() + (dy_d * torch.exp(-dy_i)).mean()  # scalar
