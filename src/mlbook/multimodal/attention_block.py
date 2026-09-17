"""A small, explicit multi-head self-attention block and a pre-norm Transformer
block, self-contained for Part VIII (vision Transformers, DETR, CLIP, VLMs).

Conventions: inputs are ``(B, N, d)`` — ``B`` batch, ``N`` tokens, ``d`` model
width. Q, K and V are three separate ``nn.Linear`` layers, never a fused
``3*d`` projection. The block implements

    Attn(X) = softmax(Q K^T / sqrt(d_head) + M) V,   Q = X W_Q, K = X W_K, V = X W_V

per head, then concatenates heads and applies ``W_O``. ``M`` is an optional
additive mask (``0`` keep, ``-inf`` / large negative drop).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with an optional additive mask.

    Input ``x``: (B, N, d). Optional ``mask``: broadcastable to (B, H, N, N),
    additive (0 = keep, -1e9 = drop). Output: (B, N, d).
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.w_o = nn.Linear(d, d)

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, N, d) -> (B, H, N, d_head)."""
        B, N, _ = t.shape
        t = t.view(B, N, self.n_heads, self.d_head)  # (B, N, H, d_head)
        return t.transpose(1, 2)  # (B, H, N, d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.split_heads(self.w_q(x))  # (B, H, N, d_head)
        k = self.split_heads(self.w_k(x))  # (B, H, N, d_head)
        v = self.split_heads(self.w_v(x))  # (B, H, N, d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, N, N)
        if mask is not None:
            scores = scores + mask  # (B, H, N, N), broadcast
        attn = torch.softmax(scores, dim=-1)  # (B, H, N, N)
        out = attn @ v  # (B, H, N, d_head)
        out = out.transpose(1, 2).reshape(B, N, self.d)  # (B, N, d)
        return self.w_o(out)  # (B, N, d)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: queries from ``x`` (B, N_q, d), keys/values from ``ctx`` (B, N_kv, d_ctx).

    Output (B, N_q, d). Used by DETR decoders, Perceiver resamplers and
    Flamingo-style gated adapters.
    """

    def __init__(self, d: int, d_ctx: int, n_heads: int) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d_ctx, d)
        self.w_v = nn.Linear(d_ctx, d)
        self.w_o = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = x.shape
        Nkv = ctx.shape[1]
        q = self.w_q(x).view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N_q, d_head)
        k = self.w_k(ctx).view(B, Nkv, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N_kv, d_head)
        v = self.w_v(ctx).view(B, Nkv, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N_kv, d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, N_q, N_kv)
        attn = torch.softmax(scores, dim=-1)  # (B, H, N_q, N_kv)
        out = attn @ v  # (B, H, N_q, d_head)
        out = out.transpose(1, 2).reshape(B, Nq, self.d)  # (B, N_q, d)
        return self.w_o(out)  # (B, N_q, d)


class MLP(nn.Module):
    """Position-wise MLP: (B, N, d) -> (B, N, d) through a ``mlp_ratio * d`` hidden layer with GELU."""

    def __init__(self, d: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))  # (B, N, mlp_ratio*d)
        return self.fc2(h)  # (B, N, d)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)); then x + MLP(LN(x)). (B, N, d) -> (B, N, d)."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_ratio)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)  # (B, N, d)
        x = x + self.mlp(self.ln2(x))  # (B, N, d)
        return x
