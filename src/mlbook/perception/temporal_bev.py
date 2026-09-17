"""Temporal BEV: ego-motion-aligned warping of the previous BEV and temporal self-attention.

The previous frame's BEV features are expressed in the *previous* ego frame.  To fuse them
with the current frame we resample them at the locations the current cells occupied then:

    p_prev = T_prev_from_curr · p_curr = (T_curr_from_prev)^{-1} · p_curr,

then ``grid_sample`` the previous BEV at ``p_prev`` (bilinear).  Cells that map outside the
previous extent read zeros.  This is the "alignment" step in BEVFormer's temporal
self-attention and in BEVDet4D; static structure lines up, moving objects leave a trail
that the attention learns to interpret as velocity.

Layout: BEV tensors are ``(B, C, X, Y)`` with X = ego forward (grid_sample's H axis) and
Y = ego left (grid_sample's W axis).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def bev_cell_centres(bev_extent: tuple[float, float, float, float], bev_hw: tuple[int, int]) -> torch.Tensor:
    """(X, Y, 2) metric (x, y) centre of every cell."""
    x_min, x_max, y_min, y_max = bev_extent
    nx, ny = bev_hw
    xs = x_min + (torch.arange(nx, dtype=torch.float32) + 0.5) * (x_max - x_min) / nx  # (X,)
    ys = y_min + (torch.arange(ny, dtype=torch.float32) + 0.5) * (y_max - y_min) / ny  # (Y,)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")  # (X, Y) each
    return torch.stack([gx, gy], dim=-1)  # (X, Y, 2)


def warp_bev(prev_bev: torch.Tensor, T_curr_from_prev: torch.Tensor,
             bev_extent: tuple[float, float, float, float]) -> torch.Tensor:
    """Resample the previous BEV so that it is expressed in the current ego frame.

    Args:
        prev_bev: (B, C, X, Y) in the previous ego frame.
        T_curr_from_prev: (B, 4, 4) rigid transform (planar motion assumed: z ignored).
    Returns:
        (B, C, X, Y) in the current ego frame; out-of-range cells are zero.
    """
    b, c, nx, ny = prev_bev.shape
    x_min, x_max, y_min, y_max = bev_extent
    centres = bev_cell_centres(bev_extent, (nx, ny)).to(prev_bev.device)  # (X, Y, 2)
    T_prev_from_curr = torch.linalg.inv(T_curr_from_prev)  # (B, 4, 4)
    R = T_prev_from_curr[:, :2, :2]  # (B, 2, 2) planar rotation
    t = T_prev_from_curr[:, :2, 3]  # (B, 2)
    p_curr = centres.view(1, nx * ny, 2)  # (1, X·Y, 2)
    p_prev = torch.matmul(p_curr, R.transpose(1, 2)) + t.view(b, 1, 2)  # (B, X·Y, 2) where each cell was, in prev frame
    # Metric → normalised grid coords in [−1, 1] (align_corners=False: edges of the extent map to ±1).
    gx_norm = (p_prev[..., 1] - y_min) / (y_max - y_min) * 2.0 - 1.0  # (B, X·Y)  W axis ↔ y
    gy_norm = (p_prev[..., 0] - x_min) / (x_max - x_min) * 2.0 - 1.0  # (B, X·Y)  H axis ↔ x
    grid = torch.stack([gx_norm, gy_norm], dim=-1).view(b, nx, ny, 2)  # (B, X, Y, 2)
    return F.grid_sample(prev_bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # (B, C, X, Y)


class TemporalSelfAttention(nn.Module):
    """Each current BEV cell attends over {its own feature, the aligned previous feature}.

    forward(curr (B, C, X, Y), prev_aligned (B, C, X, Y)) → (B, C, X, Y).
    BEVFormer's version is deformable and samples a neighbourhood; this version keeps the
    essential idea — a learned, content-dependent mixing of now and aligned-then — with an
    explicit two-key softmax so the weights are inspectable.
    """

    def __init__(self, c: int):
        super().__init__()
        self.q_proj = nn.Linear(c, c)
        self.k_proj = nn.Linear(c, c)
        self.v_proj = nn.Linear(c, c)
        self.out_proj = nn.Linear(c, c)
        self.scale = c ** -0.5

    def forward(self, curr: torch.Tensor, prev_aligned: torch.Tensor) -> torch.Tensor:
        b, c, nx, ny = curr.shape
        cur_tok = curr.permute(0, 2, 3, 1).reshape(b, nx * ny, c)  # (B, X·Y, C)
        prev_tok = prev_aligned.permute(0, 2, 3, 1).reshape(b, nx * ny, c)  # (B, X·Y, C)
        keys = torch.stack([cur_tok, prev_tok], dim=2)  # (B, X·Y, 2, C)  two candidates per cell
        q = self.q_proj(cur_tok)  # (B, X·Y, C)
        k = self.k_proj(keys)  # (B, X·Y, 2, C)
        v = self.v_proj(keys)  # (B, X·Y, 2, C)
        scores = (k * q.unsqueeze(2)).sum(-1) * self.scale  # (B, X·Y, 2)
        attn = torch.softmax(scores, dim=-1)  # (B, X·Y, 2)
        mixed = (attn.unsqueeze(-1) * v).sum(2)  # (B, X·Y, C)
        out = cur_tok + self.out_proj(mixed)  # (B, X·Y, C) residual
        return out.view(b, nx, ny, c).permute(0, 3, 1, 2).contiguous()  # (B, C, X, Y)


class TemporalBEVFusion(nn.Module):
    """warp(prev) → temporal self-attention with curr.  Keeps the recurrent state outside."""

    def __init__(self, c: int, bev_extent: tuple[float, float, float, float]):
        super().__init__()
        self.bev_extent = bev_extent
        self.attn = TemporalSelfAttention(c)

    def forward(self, curr: torch.Tensor, prev: torch.Tensor | None, T_curr_from_prev: torch.Tensor | None) -> torch.Tensor:
        """curr (B, C, X, Y); prev (B, C, X, Y) or None on the first frame → (B, C, X, Y)."""
        if prev is None:
            prev_aligned = curr  # first frame: attend over two copies of itself (BEVFormer's trick)
        else:
            prev_aligned = warp_bev(prev, T_curr_from_prev, self.bev_extent)  # (B, C, X, Y)
        return self.attn(curr, prev_aligned)  # (B, C, X, Y)
