# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/lift_splat.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k lift_splat -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/lift_splat --force

"""Lift-Splat-Shoot (Philion & Fidler, ECCV 2020) in a few explicit steps (PyTorch).

For every camera pixel the network predicts a categorical distribution over ``D`` depth
bins, ``α ∈ Δ^D``, and a context vector ``c ∈ R^C``.  The **lift** is the outer product

    f[d] = α_d · c            (D, C) per pixel,

which places a copy of the context at every depth, weighted by how likely that depth is.
The frustum point ``(u, v, d)`` is unprojected through ``K`` and ``T_ego_from_cam`` to an
ego-frame location; the **splat** sums (pillar-pools) every lifted feature that falls into
the same BEV cell.  The result is a dense BEV tensor ``(B, C, X, Y)`` that any 2D head can
consume ("shoot").

Layout: BEV tensors are ``(B, C, X, Y)`` with X = ego forward and Y = ego left.
"""
from __future__ import annotations
import torch
from torch import nn

def create_frustum(image_hw: tuple[int, int], feature_hw: tuple[int, int], depth_bins: torch.Tensor) -> torch.Tensor:
    """Pixel-space frustum ``(u, v, d)`` for every feature-map cell and depth bin.

    Args:
        image_hw: full image (H, W); feature_hw: (Hf, Wf) of the backbone output.
        depth_bins: (D,) metric depths of the bin centres.
    Returns:
        (D, Hf, Wf, 3) with entries (u, v, d) in pixels / metres.  Feature cell (i, j) is
        placed at the centre of the image patch it covers.
    """
    raise NotImplementedError('TODO: implement create_frustum (see the reference in src/mlbook)')

def frustum_to_ego(frustum: torch.Tensor, K: torch.Tensor, T_ego_from_cam: torch.Tensor) -> torch.Tensor:
    """Unproject a pixel frustum into the ego frame for ``N`` cameras.

    ``x_cam = (u − c_x) d / f_x,  y_cam = (v − c_y) d / f_y,  z_cam = d``, then
    ``p_ego = R p_cam + t`` with ``[R t] = T_ego_from_cam``.

    Args:
        frustum: (D, Hf, Wf, 3) from ``create_frustum``.
        K: (N, 3, 3) intrinsics.  T_ego_from_cam: (N, 4, 4).
    Returns:
        (N, D, Hf, Wf, 3) ego-frame xyz.
    """
    raise NotImplementedError('TODO: implement frustum_to_ego (see the reference in src/mlbook)')

def pillar_pool(features: torch.Tensor, points_ego: torch.Tensor, bev_extent: tuple[float, float, float, float], bev_hw: tuple[int, int]) -> torch.Tensor:
    """Splat lifted features into a BEV grid by summing everything that lands in a cell.

    Args:
        features: (B, N, D, Hf, Wf, C) lifted features.
        points_ego: (N, D, Hf, Wf, 3) ego xyz of each frustum point (shared over the batch).
        bev_extent: (x_min, x_max, y_min, y_max) metres.  bev_hw: (X, Y) cells.
    Returns:
        (B, C, X, Y).  Points outside the extent are dropped.  Differentiable in ``features``.
    """
    raise NotImplementedError('TODO: implement pillar_pool (see the reference in src/mlbook)')

class LiftSplat(nn.Module):
    """Toy LSS: per-camera depth distribution × context → frustum features → BEV.

    Input image features (B, N, C_in, Hf, Wf) (already extracted by a backbone) →
    output BEV features (B, C, X, Y).
    """

    def __init__(self, c_in: int, c_out: int, depth_bins: torch.Tensor, image_hw: tuple[int, int], feature_hw: tuple[int, int], bev_extent: tuple[float, float, float, float], bev_hw: tuple[int, int]):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, feats: torch.Tensor, K: torch.Tensor, T_ego_from_cam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(bev (B, C, X, Y), depth_prob (B, N, D, Hf, Wf))``.

        ``K`` (N, 3, 3) and ``T_ego_from_cam`` (N, 4, 4) describe the rig (shared over the batch).
        """
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def depth_bin_targets(lidar_depth: torch.Tensor, depth_bins: torch.Tensor) -> torch.Tensor:
    """BEVDepth-style supervision: nearest-bin index of a projected LiDAR depth map.

    Args:
        lidar_depth: (B, N, Hf, Wf) metric depth (0 where no LiDAR return).
        depth_bins: (D,).
    Returns:
        (B, N, Hf, Wf) long targets, with −1 where there is no return (ignore_index).
    """
    raise NotImplementedError('TODO: implement depth_bin_targets (see the reference in src/mlbook)')
