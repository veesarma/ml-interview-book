# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/temporal_bev.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k temporal_bev -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/temporal_bev --force

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
    raise NotImplementedError('TODO: implement bev_cell_centres (see the reference in src/mlbook)')

def warp_bev(prev_bev: torch.Tensor, T_curr_from_prev: torch.Tensor, bev_extent: tuple[float, float, float, float]) -> torch.Tensor:
    """Resample the previous BEV so that it is expressed in the current ego frame.

    Args:
        prev_bev: (B, C, X, Y) in the previous ego frame.
        T_curr_from_prev: (B, 4, 4) rigid transform (planar motion assumed: z ignored).
    Returns:
        (B, C, X, Y) in the current ego frame; out-of-range cells are zero.
    """
    raise NotImplementedError('TODO: implement warp_bev (see the reference in src/mlbook)')

class TemporalSelfAttention(nn.Module):
    """Each current BEV cell attends over {its own feature, the aligned previous feature}.

    forward(curr (B, C, X, Y), prev_aligned (B, C, X, Y)) → (B, C, X, Y).
    BEVFormer's version is deformable and samples a neighbourhood; this version keeps the
    essential idea — a learned, content-dependent mixing of now and aligned-then — with an
    explicit two-key softmax so the weights are inspectable.
    """

    def __init__(self, c: int):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, curr: torch.Tensor, prev_aligned: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TemporalBEVFusion(nn.Module):
    """warp(prev) → temporal self-attention with curr.  Keeps the recurrent state outside."""

    def __init__(self, c: int, bev_extent: tuple[float, float, float, float]):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, curr: torch.Tensor, prev: torch.Tensor | None, T_curr_from_prev: torch.Tensor | None) -> torch.Tensor:
        """curr (B, C, X, Y); prev (B, C, X, Y) or None on the first frame → (B, C, X, Y)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
