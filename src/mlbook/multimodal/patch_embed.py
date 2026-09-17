"""Patchification for vision Transformers.

An image ``(B, C, H, W)`` is cut into ``N = (H/P) * (W/P)`` non-overlapping
``P x P`` patches; each patch is flattened to a ``P*P*C`` vector and mapped to
``R^d`` by ONE shared linear map ``E in R^{(P*P*C) x d}``. That linear map is
exactly a convolution with ``kernel_size = stride = P`` — both are implemented
here and tested equal. Also: learned and 2-D sinusoidal position embeddings and
bicubic interpolation of a position grid when the input resolution changes.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, P*P*C) with N = (H/P)*(W/P), row-major patch order.

    Patch (i, j) covers rows ``i*P:(i+1)*P`` and cols ``j*P:(j+1)*P``; its
    vector is laid out as (C, P, P) flattened, matching ``nn.Conv2d`` weight order.
    """
    B, C, H, W = x.shape
    if H % patch or W % patch:
        raise ValueError("H and W must be multiples of the patch size")
    h, w = H // patch, W // patch
    x = x.view(B, C, h, patch, w, patch)  # (B, C, h, P, w, P)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, h, w, C, P, P)
    return x.reshape(B, h * w, C * patch * patch)  # (B, N, P*P*C)


def unpatchify(tokens: torch.Tensor, patch: int, channels: int, H: int, W: int) -> torch.Tensor:
    """Inverse of ``patchify``: (B, N, P*P*C) -> (B, C, H, W)."""
    B = tokens.shape[0]
    h, w = H // patch, W // patch
    x = tokens.view(B, h, w, channels, patch, patch)  # (B, h, w, C, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, P, w, P)
    return x.reshape(B, channels, H, W)  # (B, C, H, W)


class PatchEmbedLinear(nn.Module):
    """Patchify + shared linear map: (B, C, H, W) -> (B, N, d). ``z_n = flatten(patch_n) E + b``."""

    def __init__(self, in_channels: int, patch: int, d: int) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(in_channels * patch * patch, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = patchify(x, self.patch)  # (B, N, P*P*C)
        return self.proj(tokens)  # (B, N, d)


class PatchEmbedConv(nn.Module):
    """The same map as a conv with kernel = stride = P: (B, C, H, W) -> (B, N, d)."""

    def __init__(self, in_channels: int, patch: int, d: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)  # (B, d, H/P, W/P)
        return z.flatten(2).transpose(1, 2)  # (B, N, d)

    @torch.no_grad()
    def load_from_linear(self, lin: PatchEmbedLinear) -> None:
        """Copy weights from the linear version: W_lin is (d, P*P*C) -> conv weight (d, C, P, P)."""
        d, _ = lin.proj.weight.shape
        self.proj.weight.copy_(lin.proj.weight.view(self.proj.weight.shape))  # (d, C, P, P)
        self.proj.bias.copy_(lin.proj.bias)  # (d,)


def sinusoidal_2d_pos_embed(h: int, w: int, d: int) -> torch.Tensor:
    """Fixed 2-D sin/cos position embedding, (h*w, d): first d/2 dims encode the row, last d/2 the column.

    Each half uses the 1-D Transformer recipe with frequencies ``1 / 10000^{2i/(d/2)}``.
    """
    if d % 4 != 0:
        raise ValueError("d must be divisible by 4")
    half = d // 2

    def one_axis(n: int) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32)[:, None]  # (n, 1)
        i = torch.arange(half // 2, dtype=torch.float32)[None, :]  # (1, half/2)
        angle = pos / (10000 ** (2 * i / half))  # (n, half/2)
        return torch.cat([angle.sin(), angle.cos()], dim=1)  # (n, half)

    rows = one_axis(h)[:, None, :].expand(h, w, half)  # (h, w, half)
    cols = one_axis(w)[None, :, :].expand(h, w, half)  # (h, w, half)
    return torch.cat([rows, cols], dim=-1).reshape(h * w, d)  # (h*w, d)


def interpolate_pos_embed(pos: torch.Tensor, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a learned position grid: (h0*w0, d) -> (h1*w1, d) by bicubic interpolation.

    This is how a ViT pretrained at 224 px is fine-tuned at 384 px: the *grid* of
    embeddings is treated as a d-channel image and resampled.
    """
    h0, w0 = old_hw
    h1, w1 = new_hw
    d = pos.shape[1]
    grid = pos.view(1, h0, w0, d).permute(0, 3, 1, 2)  # (1, d, h0, w0)
    grid = nn.functional.interpolate(grid, size=(h1, w1), mode="bicubic", align_corners=False)  # (1, d, h1, w1)
    return grid.permute(0, 2, 3, 1).reshape(h1 * w1, d)  # (h1*w1, d)


def num_patches(H: int, W: int, patch: int) -> int:
    """N = (H/P)(W/P): the sequence length a ViT sees, so attention cost is O(N^2) = O((HW)^2 / P^4)."""
    return math.ceil(H / patch) * math.ceil(W / patch)
