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
    B, h, w, d = x.shape
    x = x.view(B, h // M, M, w // M, M, d)  # (B, h/M, M, w/M, M, d)
    x = x.permute(0, 1, 3, 2, 4, 5)  # (B, h/M, w/M, M, M, d)
    return x.reshape(-1, M * M, d)  # (B*nW, M*M, d)


def window_reverse(windows: torch.Tensor, M: int, h: int, w: int) -> torch.Tensor:
    """Inverse of ``window_partition``: (B*nW, M*M, d) -> (B, h, w, d)."""
    nW = (h // M) * (w // M)
    B = windows.shape[0] // nW
    d = windows.shape[-1]
    x = windows.view(B, h // M, w // M, M, M, d)  # (B, h/M, w/M, M, M, d)
    x = x.permute(0, 1, 3, 2, 4, 5)  # (B, h/M, M, w/M, M, d)
    return x.reshape(B, h, w, d)  # (B, h, w, d)


def region_ids(h: int, w: int, shift: int) -> torch.Tensor:
    """Label each cell of the UN-shifted grid by whether it will wrap around under the cyclic shift. (h, w) ints.

    ``torch.roll(x, -shift)`` moves rows ``[0, shift)`` to the bottom and columns ``[0, shift)``
    to the right edge. A wrapped cell is not a spatial neighbour of the cells it lands next
    to, so the id is ``2 * row_wrapped + col_wrapped`` (four regions). The official Swin code
    labels 3 x 3 = 9 regions in rolled coordinates; inside any window the two labellings
    induce the same partition because the extra boundary sits exactly on a window edge.
    """
    ids = torch.zeros(h, w, dtype=torch.long)  # (h, w)
    ids[:shift, :] += 2  # rows that wrap to the bottom
    ids[:, :shift] += 1  # columns that wrap to the right
    return ids


def shifted_window_mask(h: int, w: int, M: int, shift: int) -> torch.Tensor:
    """Additive attention mask for shifted windows: (nW, M*M, M*M), 0 keep / -100 drop.

    Build region ids on the un-shifted grid, roll them exactly as the tokens
    are rolled, partition into windows, and mark pairs (q, k) whose ids differ.
    Requires h, w >= 2M so a wrapped cell is never a true neighbour of an unwrapped one.
    """
    if shift == 0:
        return torch.zeros((h // M) * (w // M), M * M, M * M)  # (nW, M*M, M*M)
    ids = region_ids(h, w, shift)  # (h, w)
    ids = torch.roll(ids, shifts=(-shift, -shift), dims=(0, 1))  # (h, w), same roll as tokens
    win = window_partition(ids[None, :, :, None].float(), M).squeeze(-1)  # (nW, M*M)
    diff = win[:, :, None] - win[:, None, :]  # (nW, M*M, M*M)
    return torch.where(diff == 0, 0.0, -100.0)  # (nW, M*M, M*M)


def relative_position_index(M: int) -> torch.Tensor:
    """For each (query, key) pair in an M x M window, the index into a (2M-1)^2 bias table. (M*M, M*M) long."""
    coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij"))  # (2, M, M)
    coords = coords.flatten(1)  # (2, M*M)
    rel = coords[:, :, None] - coords[:, None, :]  # (2, M*M, M*M), values in [-(M-1), M-1]
    rel = rel.permute(1, 2, 0).contiguous()  # (M*M, M*M, 2)
    rel[:, :, 0] += M - 1  # shift rows to [0, 2M-2]
    rel[:, :, 1] += M - 1  # shift cols to [0, 2M-2]
    return rel[:, :, 0] * (2 * M - 1) + rel[:, :, 1]  # (M*M, M*M)


class WindowAttention(nn.Module):
    """Multi-head attention inside windows with relative position bias.

    Input ``x``: (B*nW, M*M, d); optional ``mask``: (nW, M*M, M*M) additive.
    Output: (B*nW, M*M, d).  logits = QK^T/sqrt(d_head) + Bias[rel_idx] + mask.
    """

    def __init__(self, d: int, n_heads: int, M: int) -> None:
        super().__init__()
        self.d, self.n_heads, self.d_head, self.M = d, n_heads, d // n_heads, M
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.w_o = nn.Linear(d, d)
        self.bias_table = nn.Parameter(torch.zeros((2 * M - 1) ** 2, n_heads))  # ((2M-1)^2, H)
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        self.register_buffer("rel_index", relative_position_index(M), persistent=False)  # (M*M, M*M)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        BnW, L, _ = x.shape
        q = self.w_q(x).view(BnW, L, self.n_heads, self.d_head).transpose(1, 2)  # (B*nW, H, L, d_head)
        k = self.w_k(x).view(BnW, L, self.n_heads, self.d_head).transpose(1, 2)  # (B*nW, H, L, d_head)
        v = self.w_v(x).view(BnW, L, self.n_heads, self.d_head).transpose(1, 2)  # (B*nW, H, L, d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B*nW, H, L, L)
        bias = self.bias_table[self.rel_index.view(-1)].view(L, L, self.n_heads)  # (L, L, H)
        scores = scores + bias.permute(2, 0, 1)[None]  # (B*nW, H, L, L)
        if mask is not None:
            nW = mask.shape[0]
            scores = scores.view(BnW // nW, nW, self.n_heads, L, L) + mask[None, :, None]  # (B, nW, H, L, L)
            scores = scores.view(BnW, self.n_heads, L, L)  # (B*nW, H, L, L)
        attn = torch.softmax(scores, dim=-1)  # (B*nW, H, L, L)
        out = (attn @ v).transpose(1, 2).reshape(BnW, L, self.d)  # (B*nW, L, d)
        return self.w_o(out)  # (B*nW, L, d)


class SwinBlock(nn.Module):
    """One (shifted-)window attention block on a token grid: (B, h, w, d) -> (B, h, w, d).

    ``shift=0`` gives W-MSA; ``shift=M//2`` gives SW-MSA (roll, attend with mask, roll back).
    """

    def __init__(self, d: int, n_heads: int, M: int, shift: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.M, self.shift = M, shift
        self.ln1 = nn.LayerNorm(d)
        self.attn = WindowAttention(d, n_heads, M)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, h, w, d = x.shape
        y = self.ln1(x)  # (B, h, w, d)
        if self.shift > 0:
            y = torch.roll(y, shifts=(-self.shift, -self.shift), dims=(1, 2))  # (B, h, w, d)
            mask = shifted_window_mask(h, w, self.M, self.shift).to(x.device)  # (nW, M*M, M*M)
        else:
            mask = None
        y = window_partition(y, self.M)  # (B*nW, M*M, d)
        y = self.attn(y, mask)  # (B*nW, M*M, d)
        y = window_reverse(y, self.M, h, w)  # (B, h, w, d)
        if self.shift > 0:
            y = torch.roll(y, shifts=(self.shift, self.shift), dims=(1, 2))  # (B, h, w, d)
        x = x + y  # (B, h, w, d)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))  # (B, h, w, d)
        return x


class PatchMerging(nn.Module):
    """Swin downsampling: (B, h, w, d) -> (B, h/2, w/2, 2d). Concatenate each 2x2 group (4d), LN, Linear(4d -> 2d)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * d)
        self.reduce = nn.Linear(4 * d, 2 * d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]  # (B, h/2, w/2, d)
        x1 = x[:, 1::2, 0::2, :]  # (B, h/2, w/2, d)
        x2 = x[:, 0::2, 1::2, :]  # (B, h/2, w/2, d)
        x3 = x[:, 1::2, 1::2, :]  # (B, h/2, w/2, d)
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, h/2, w/2, 4d)
        return self.reduce(self.norm(x))  # (B, h/2, w/2, 2d)
