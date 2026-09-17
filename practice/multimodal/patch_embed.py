# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/patch_embed.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k patch_embed -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/patch_embed --force

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
    raise NotImplementedError('TODO: implement patchify (see the reference in src/mlbook)')

def unpatchify(tokens: torch.Tensor, patch: int, channels: int, H: int, W: int) -> torch.Tensor:
    """Inverse of ``patchify``: (B, N, P*P*C) -> (B, C, H, W)."""
    raise NotImplementedError('TODO: implement unpatchify (see the reference in src/mlbook)')

class PatchEmbedLinear(nn.Module):
    """Patchify + shared linear map: (B, C, H, W) -> (B, N, d). ``z_n = flatten(patch_n) E + b``."""

    def __init__(self, in_channels: int, patch: int, d: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class PatchEmbedConv(nn.Module):
    """The same map as a conv with kernel = stride = P: (B, C, H, W) -> (B, N, d)."""

    def __init__(self, in_channels: int, patch: int, d: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    @torch.no_grad()
    def load_from_linear(self, lin: PatchEmbedLinear) -> None:
        """Copy weights from the linear version: W_lin is (d, P*P*C) -> conv weight (d, C, P, P)."""
        raise NotImplementedError('TODO: implement load_from_linear (see the reference in src/mlbook)')

def sinusoidal_2d_pos_embed(h: int, w: int, d: int) -> torch.Tensor:
    """Fixed 2-D sin/cos position embedding, (h*w, d): first d/2 dims encode the row, last d/2 the column.

    Each half uses the 1-D Transformer recipe with frequencies ``1 / 10000^{2i/(d/2)}``.
    """
    raise NotImplementedError('TODO: implement sinusoidal_2d_pos_embed (see the reference in src/mlbook)')

def interpolate_pos_embed(pos: torch.Tensor, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a learned position grid: (h0*w0, d) -> (h1*w1, d) by bicubic interpolation.

    This is how a ViT pretrained at 224 px is fine-tuned at 384 px: the *grid* of
    embeddings is treated as a d-channel image and resampled.
    """
    raise NotImplementedError('TODO: implement interpolate_pos_embed (see the reference in src/mlbook)')

def num_patches(H: int, W: int, patch: int) -> int:
    """N = (H/P)(W/P): the sequence length a ViT sees, so attention cost is O(N^2) = O((HW)^2 / P^4)."""
    raise NotImplementedError('TODO: implement num_patches (see the reference in src/mlbook)')
