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
    h, w = image_hw
    hf, wf = feature_hw
    d = depth_bins.numel()
    us = (torch.arange(wf, dtype=torch.float32) + 0.5) * (w / wf)  # (Wf,) pixel u of each column
    vs = (torch.arange(hf, dtype=torch.float32) + 0.5) * (h / hf)  # (Hf,) pixel v of each row
    u_grid = us.view(1, 1, wf).expand(d, hf, wf)  # (D, Hf, Wf)
    v_grid = vs.view(1, hf, 1).expand(d, hf, wf)  # (D, Hf, Wf)
    d_grid = depth_bins.view(d, 1, 1).expand(d, hf, wf)  # (D, Hf, Wf)
    return torch.stack([u_grid, v_grid, d_grid], dim=-1)  # (D, Hf, Wf, 3)


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
    u = frustum[..., 0]  # (D, Hf, Wf)
    v = frustum[..., 1]  # (D, Hf, Wf)
    d = frustum[..., 2]  # (D, Hf, Wf)
    fx = K[:, 0, 0].view(-1, 1, 1, 1)  # (N, 1, 1, 1)
    fy = K[:, 1, 1].view(-1, 1, 1, 1)  # (N, 1, 1, 1)
    cx = K[:, 0, 2].view(-1, 1, 1, 1)  # (N, 1, 1, 1)
    cy = K[:, 1, 2].view(-1, 1, 1, 1)  # (N, 1, 1, 1)
    x_cam = (u.unsqueeze(0) - cx) * d.unsqueeze(0) / fx  # (N, D, Hf, Wf)
    y_cam = (v.unsqueeze(0) - cy) * d.unsqueeze(0) / fy  # (N, D, Hf, Wf)
    z_cam = d.unsqueeze(0).expand_as(x_cam)  # (N, D, Hf, Wf)
    p_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, D, Hf, Wf, 3)
    R = T_ego_from_cam[:, :3, :3]  # (N, 3, 3)
    t = T_ego_from_cam[:, :3, 3]  # (N, 3)
    # Row-vector convention: p_ego = p_cam @ R^T + t, broadcast over (D, Hf, Wf).
    p_ego = torch.matmul(p_cam, R.transpose(1, 2).view(-1, 1, 1, 3, 3)) + t.view(-1, 1, 1, 1, 3)  # (N, D, Hf, Wf, 3)
    return p_ego


def pillar_pool(features: torch.Tensor, points_ego: torch.Tensor, bev_extent: tuple[float, float, float, float],
                bev_hw: tuple[int, int]) -> torch.Tensor:
    """Splat lifted features into a BEV grid by summing everything that lands in a cell.

    Args:
        features: (B, N, D, Hf, Wf, C) lifted features.
        points_ego: (N, D, Hf, Wf, 3) ego xyz of each frustum point (shared over the batch).
        bev_extent: (x_min, x_max, y_min, y_max) metres.  bev_hw: (X, Y) cells.
    Returns:
        (B, C, X, Y).  Points outside the extent are dropped.  Differentiable in ``features``.
    """
    b, n, d, hf, wf, c = features.shape
    x_min, x_max, y_min, y_max = bev_extent
    nx, ny = bev_hw
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    ix = torch.floor((points_ego[..., 0] - x_min) / dx).long()  # (N, D, Hf, Wf)
    iy = torch.floor((points_ego[..., 1] - y_min) / dy).long()  # (N, D, Hf, Wf)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)  # (N, D, Hf, Wf)
    cell = (ix * ny + iy).view(1, -1).expand(b, -1)  # (B, N·D·Hf·Wf) flat cell index per point
    batch_offset = (torch.arange(b) * nx * ny).view(b, 1)  # (B, 1)
    flat_index = (cell + batch_offset)[:, inside.view(-1)].reshape(-1)  # (M,) kept points across batch
    flat_feats = features.reshape(b, -1, c)[:, inside.view(-1)].reshape(-1, c)  # (M, C)
    bev = torch.zeros(b * nx * ny, c, dtype=features.dtype, device=features.device)  # (B·X·Y, C)
    bev = bev.index_add(0, flat_index, flat_feats)  # (B·X·Y, C) sum-pool per cell
    return bev.view(b, nx, ny, c).permute(0, 3, 1, 2).contiguous()  # (B, C, X, Y)


class LiftSplat(nn.Module):
    """Toy LSS: per-camera depth distribution × context → frustum features → BEV.

    Input image features (B, N, C_in, Hf, Wf) (already extracted by a backbone) →
    output BEV features (B, C, X, Y).
    """

    def __init__(self, c_in: int, c_out: int, depth_bins: torch.Tensor, image_hw: tuple[int, int],
                 feature_hw: tuple[int, int], bev_extent: tuple[float, float, float, float], bev_hw: tuple[int, int]):
        super().__init__()
        self.c_out = c_out
        self.d = depth_bins.numel()
        self.bev_extent = bev_extent
        self.bev_hw = bev_hw
        self.depth_head = nn.Conv2d(c_in, self.d, kernel_size=1)  # logits over depth bins
        self.context_head = nn.Conv2d(c_in, c_out, kernel_size=1)  # context vector per pixel
        self.register_buffer("frustum", create_frustum(image_hw, feature_hw, depth_bins))  # (D, Hf, Wf, 3)

    def forward(self, feats: torch.Tensor, K: torch.Tensor, T_ego_from_cam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(bev (B, C, X, Y), depth_prob (B, N, D, Hf, Wf))``.

        ``K`` (N, 3, 3) and ``T_ego_from_cam`` (N, 4, 4) describe the rig (shared over the batch).
        """
        b, n, c_in, hf, wf = feats.shape
        flat = feats.reshape(b * n, c_in, hf, wf)  # (B·N, C_in, Hf, Wf)
        depth_prob = torch.softmax(self.depth_head(flat), dim=1)  # (B·N, D, Hf, Wf)  α per pixel
        context = self.context_head(flat)  # (B·N, C, Hf, Wf)
        lifted = depth_prob.unsqueeze(2) * context.unsqueeze(1)  # (B·N, D, C, Hf, Wf)  outer product α_d · c
        lifted = lifted.permute(0, 1, 3, 4, 2).reshape(b, n, self.d, hf, wf, self.c_out)  # (B, N, D, Hf, Wf, C)
        points_ego = frustum_to_ego(self.frustum, K, T_ego_from_cam)  # (N, D, Hf, Wf, 3)
        bev = pillar_pool(lifted, points_ego, self.bev_extent, self.bev_hw)  # (B, C, X, Y)
        return bev, depth_prob.view(b, n, self.d, hf, wf)


def depth_bin_targets(lidar_depth: torch.Tensor, depth_bins: torch.Tensor) -> torch.Tensor:
    """BEVDepth-style supervision: nearest-bin index of a projected LiDAR depth map.

    Args:
        lidar_depth: (B, N, Hf, Wf) metric depth (0 where no LiDAR return).
        depth_bins: (D,).
    Returns:
        (B, N, Hf, Wf) long targets, with −1 where there is no return (ignore_index).
    """
    diff = (lidar_depth.unsqueeze(-1) - depth_bins.view(1, 1, 1, 1, -1)).abs()  # (B, N, Hf, Wf, D)
    target = diff.argmin(dim=-1)  # (B, N, Hf, Wf)
    return torch.where(lidar_depth > 0, target, torch.full_like(target, -1))  # (B, N, Hf, Wf)
