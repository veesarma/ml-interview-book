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
    B, C, T, H, W = video.shape
    x = video.view(B, C, T // t, t, H // P, P, W // P, P)  # (B, C, T', t, h, P, w, P)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, T', h, w, C, t, P, P)
    return x.reshape(B, T // t, H // P, W // P, C * t * P * P)  # (B, T', h, w, t*P*P*C)


class TubeletEmbed(nn.Module):
    """Tubelet flatten + shared linear map: (B, C, T, H, W) -> (B, T', h, w, d)."""

    def __init__(self, in_channels: int, t: int, P: int, d: int) -> None:
        super().__init__()
        self.t, self.P = t, P
        self.proj = nn.Linear(in_channels * t * P * P, d)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.proj(tubelet_embed(video, self.t, self.P))  # (B, T', h, w, d)


class FactorisedSpaceTimeBlock(nn.Module):
    """Spatial attention per frame, then temporal attention per location. (B, T', h, w, d) -> same.

    x = x + SpatialAttn(LN(x))   (attention over the h*w tokens of each frame)
    x = x + TemporalAttn(LN(x))  (attention over the T' tokens at each (i, j))
    x = x + MLP(LN(x))
    """

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln_s = nn.LayerNorm(d)
        self.spatial = MultiHeadSelfAttention(d, n_heads)
        self.ln_t = nn.LayerNorm(d)
        self.temporal = MultiHeadSelfAttention(d, n_heads)
        self.ln_m = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, h, w, d = x.shape
        s = self.ln_s(x).reshape(B * T, h * w, d)  # (B*T', h*w, d): one sequence per frame
        s = self.spatial(s).view(B, T, h, w, d)  # (B, T', h, w, d)
        x = x + s
        t = self.ln_t(x).permute(0, 2, 3, 1, 4).reshape(B * h * w, T, d)  # (B*h*w, T', d): one sequence per location
        t = self.temporal(t).view(B, h, w, T, d).permute(0, 3, 1, 2, 4)  # (B, T', h, w, d)
        x = x + t
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln_m(x))))  # (B, T', h, w, d)
        return x


class JointSpaceTimeBlock(nn.Module):
    """Full attention over all T'*h*w tokens: (B, T', h, w, d) -> same. The O(N^2) baseline."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln_a = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, n_heads)
        self.ln_m = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, h, w, d = x.shape
        a = self.ln_a(x).reshape(B, T * h * w, d)  # (B, N, d)
        x = x + self.attn(a).view(B, T, h, w, d)  # (B, T', h, w, d)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln_m(x))))  # (B, T', h, w, d)
        return x


def attention_flops(T: int, h: int, w: int, d: int, factorised: bool) -> int:
    """Multiply-adds in the QK^T and AV products only (the O(N^2) part), per example.

    Joint:       2 * N^2 * d,             N = T*h*w
    Factorised:  2 * T*(hw)^2 * d  +  2 * hw * T^2 * d
    """
    S = h * w
    if factorised:
        return 2 * T * S * S * d + 2 * S * T * T * d
    N = T * S
    return 2 * N * N * d
