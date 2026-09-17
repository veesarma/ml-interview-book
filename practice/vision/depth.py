# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/vision/depth.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k depth -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py vision/depth --force

"""Depth from stereo and monocular self-supervision: disparity ↔ depth, block matching,
cost volumes, and the photometric (SSIM + L1) reconstruction loss.

Stereo geometry (rectified pair, baseline ``b`` metres, focal ``f`` pixels):

    disparity d = x_L − x_R = f · b / Z      ⇔      Z = f · b / d.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def disparity_to_depth(disparity: np.ndarray, focal_px: float, baseline_m: float, eps: float=1e-06) -> np.ndarray:
    """``Z = f·b / d``; zero disparity maps to ``inf`` (point at infinity).  Any shape."""
    raise NotImplementedError('TODO: implement disparity_to_depth (see the reference in src/mlbook)')

def depth_to_disparity(depth: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    """``d = f·b / Z``.  Any shape."""
    raise NotImplementedError('TODO: implement depth_to_disparity (see the reference in src/mlbook)')

def depth_error_from_disparity_error(depth: np.ndarray, focal_px: float, baseline_m: float, disp_err_px: float) -> np.ndarray:
    """First-order depth error ``|ΔZ| ≈ Z² · Δd / (f·b)`` — error grows *quadratically* with range."""
    raise NotImplementedError('TODO: implement depth_error_from_disparity_error (see the reference in src/mlbook)')

def stereo_block_matching(left: np.ndarray, right: np.ndarray, max_disparity: int, block: int=5) -> np.ndarray:
    """Winner-take-all SAD block matching on a rectified grayscale pair.

    For each candidate ``d`` build the cost ``C[d, i, j] = Σ_window |L(i, j) − R(i, j − d)|``
    (a 3-D *cost volume*), then take ``argmin_d``.  Left border columns where ``j − d < 0``
    get maximal cost.

    Args:
        left, right: (H, W) float images.
    Returns:
        (H, W) integer disparities in [0, max_disparity).
    """
    raise NotImplementedError('TODO: implement stereo_block_matching (see the reference in src/mlbook)')

def build_cost_volume(feat_left: torch.Tensor, feat_right: torch.Tensor, max_disparity: int) -> torch.Tensor:
    """Concatenation cost volume (PSMNet-style) for 3-D conv aggregation.

    Args:
        feat_left, feat_right: (B, C, H, W) feature maps.
    Returns:
        (B, 2C, D, H, W): at disparity ``d`` the right features are shifted right by ``d``.
    """
    raise NotImplementedError('TODO: implement build_cost_volume (see the reference in src/mlbook)')

def soft_argmin_disparity(cost: torch.Tensor) -> torch.Tensor:
    """Differentiable disparity ``d̂ = Σ_d d · softmax(−cost)_d`` (GC-Net).  ``cost``: (B, D, H, W) → (B, H, W)."""
    raise NotImplementedError('TODO: implement soft_argmin_disparity (see the reference in src/mlbook)')

def warp_right_to_left(right: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    """Reconstruct the left image by sampling the right at ``x − d`` with bilinear ``grid_sample``.

    Args:
        right: (B, C, H, W).  disparity: (B, H, W) in pixels (positive = shift left).
    Returns:
        (B, C, H, W) synthesised left view.
    """
    raise NotImplementedError('TODO: implement warp_right_to_left (see the reference in src/mlbook)')

def ssim(x: torch.Tensor, y: torch.Tensor, c1: float=0.01 ** 2, c2: float=0.03 ** 2) -> torch.Tensor:
    """Per-pixel SSIM with a 3×3 average window (as in Monodepth).  Inputs (B, C, H, W) → (B, C, H, W)."""
    raise NotImplementedError('TODO: implement ssim (see the reference in src/mlbook)')

def photometric_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float=0.85) -> torch.Tensor:
    """Monodepth appearance loss ``α (1−SSIM)/2 + (1−α) |pred − target|`` averaged. Inputs (B, C, H, W)."""
    raise NotImplementedError('TODO: implement photometric_loss (see the reference in src/mlbook)')

def smoothness_loss(disparity: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware smoothness ``|∂x d| e^{−|∂x I|} + |∂y d| e^{−|∂y I|}``: smooth except at image edges.

    Args:
        disparity: (B, H, W) (use mean-normalised disparity).  image: (B, C, H, W).
    """
    raise NotImplementedError('TODO: implement smoothness_loss (see the reference in src/mlbook)')
