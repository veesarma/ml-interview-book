# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/video_attention.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k video_attention -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/video_attention --force

"""Video tokens and factorised space-time attention.

A clip ``(B, C, T, H, W)`` is cut into tubelets of size ``(t, P, P)`` giving
``N = (T/t) * (H/P) * (W/P)`` tokens, kept here as a grid ``(B, T', h, w, d)``.

Joint attention: every token attends to all N tokens          -> O(N^2 d) = O((T' h w)^2 d)
Factorised:  spatial attention within each frame (h*w tokens) -> O(T' (hw)^2 d)
             then temporal attention along each pixel track   -> O(hw T'^2 d)
"""
from __future__ import annotations
import torch
from torch import nn
from .attention_block import MultiHeadSelfAttention

def tubelet_embed(video: torch.Tensor, t: int, P: int) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, T/t, H/P, W/P, t*P*P*C): flatten each tubelet (ViViT 'tubelet embedding')."""
    raise NotImplementedError('TODO: implement tubelet_embed (see the reference in src/mlbook)')

class TubeletEmbed(nn.Module):
    """Tubelet flatten + shared linear map: (B, C, T, H, W) -> (B, T', h, w, d)."""

    def __init__(self, in_channels: int, t: int, P: int, d: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class FactorisedSpaceTimeBlock(nn.Module):
    """Spatial attention per frame, then temporal attention per location. (B, T', h, w, d) -> same.

    x = x + SpatialAttn(LN(x))   (attention over the h*w tokens of each frame)
    x = x + TemporalAttn(LN(x))  (attention over the T' tokens at each (i, j))
    x = x + MLP(LN(x))
    """

    def __init__(self, d: int, n_heads: int, mlp_ratio: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class JointSpaceTimeBlock(nn.Module):
    """Full attention over all T'*h*w tokens: (B, T', h, w, d) -> same. The O(N^2) baseline."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def attention_flops(T: int, h: int, w: int, d: int, factorised: bool) -> int:
    """Multiply-adds in the QK^T and AV products only (the O(N^2) part), per example.

    Joint:       2 * N^2 * d,             N = T*h*w
    Factorised:  2 * T*(hw)^2 * d  +  2 * hw * T^2 * d
    """
    raise NotImplementedError('TODO: implement attention_flops (see the reference in src/mlbook)')
