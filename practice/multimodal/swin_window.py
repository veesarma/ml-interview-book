# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/swin_window.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k swin_window -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/swin_window --force

"""Swin-style windowed and shifted-window attention, with the cyclic-shift mask.

Tokens live on an ``h x w`` grid, ``(B, h, w, d)``. Windowed attention runs
full attention inside each non-overlapping ``M x M`` window — cost
``O(h*w * M^2 * d)`` instead of ``O((h*w)^2 * d)``. Shifted windows roll the
grid by ``M/2``; after the roll, one window can contain tokens from up to four
*different* original regions, so a mask blocks attention between tokens whose
region ids differ. Relative position bias adds a learned ``B[Δi, Δj]`` to the
logits, indexed by the in-window offset of the query/key pair.
"""
from __future__ import annotations
import math
import torch
from torch import nn

def window_partition(x: torch.Tensor, M: int) -> torch.Tensor:
    """(B, h, w, d) -> (B * nW, M*M, d), windows in row-major order, nW = (h/M)*(w/M)."""
    raise NotImplementedError('TODO: implement window_partition (see the reference in src/mlbook)')

def window_reverse(windows: torch.Tensor, M: int, h: int, w: int) -> torch.Tensor:
    """Inverse of ``window_partition``: (B*nW, M*M, d) -> (B, h, w, d)."""
    raise NotImplementedError('TODO: implement window_reverse (see the reference in src/mlbook)')

def region_ids(h: int, w: int, shift: int) -> torch.Tensor:
    """Label each cell of the UN-shifted grid by whether it will wrap around under the cyclic shift. (h, w) ints.

    ``torch.roll(x, -shift)`` moves rows ``[0, shift)`` to the bottom and columns ``[0, shift)``
    to the right edge. A wrapped cell is not a spatial neighbour of the cells it lands next
    to, so the id is ``2 * row_wrapped + col_wrapped`` (four regions). The official Swin code
    labels 3 x 3 = 9 regions in rolled coordinates; inside any window the two labellings
    induce the same partition because the extra boundary sits exactly on a window edge.
    """
    raise NotImplementedError('TODO: implement region_ids (see the reference in src/mlbook)')

def shifted_window_mask(h: int, w: int, M: int, shift: int) -> torch.Tensor:
    """Additive attention mask for shifted windows: (nW, M*M, M*M), 0 keep / -100 drop.

    Build region ids on the un-shifted grid, roll them exactly as the tokens
    are rolled, partition into windows, and mark pairs (q, k) whose ids differ.
    Requires h, w >= 2M so a wrapped cell is never a true neighbour of an unwrapped one.
    """
    raise NotImplementedError('TODO: implement shifted_window_mask (see the reference in src/mlbook)')

def relative_position_index(M: int) -> torch.Tensor:
    """For each (query, key) pair in an M x M window, the index into a (2M-1)^2 bias table. (M*M, M*M) long."""
    raise NotImplementedError('TODO: implement relative_position_index (see the reference in src/mlbook)')

class WindowAttention(nn.Module):
    """Multi-head attention inside windows with relative position bias.

    Input ``x``: (B*nW, M*M, d); optional ``mask``: (nW, M*M, M*M) additive.
    Output: (B*nW, M*M, d).  logits = QK^T/sqrt(d_head) + Bias[rel_idx] + mask.
    """

    def __init__(self, d: int, n_heads: int, M: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class SwinBlock(nn.Module):
    """One (shifted-)window attention block on a token grid: (B, h, w, d) -> (B, h, w, d).

    ``shift=0`` gives W-MSA; ``shift=M//2`` gives SW-MSA (roll, attend with mask, roll back).
    """

    def __init__(self, d: int, n_heads: int, M: int, shift: int, mlp_ratio: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class PatchMerging(nn.Module):
    """Swin downsampling: (B, h, w, d) -> (B, h/2, w/2, 2d). Concatenate each 2x2 group (4d), LN, Linear(4d -> 2d)."""

    def __init__(self, d: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
