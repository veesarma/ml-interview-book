"""Transformer blocks: encoder (self-attention + FFN) and decoder (self + cross + FFN),
with Pre-LN (default; GPT-2, LLaMA) or Post-LN (original Transformer, BERT) placement and
LayerNorm or RMSNorm.

Pre-LN:  x = x + Attn(LN(x));  x = x + FFN(LN(x))       (residual stream never normalised)
Post-LN: x = LN(x + Attn(x));  x = LN(x + FFN(x))       (normalisation inside the stream)
"""

from __future__ import annotations

import torch
from torch import nn

from .ffn import FeedForward, GatedFeedForward
from .kv_cache import LayerKVCache
from .multihead import CrossAttention, MultiHeadAttention


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g  -- LayerNorm without the mean subtraction (Zhang & Sennrich 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # (d_model,) gain g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B, T, 1)
        return x / rms * self.weight  # (B, T, d_model)


def make_norm(kind: str, d_model: int) -> nn.Module:
    return nn.LayerNorm(d_model) if kind == "layer" else RMSNorm(d_model)


def make_ffn(kind: str, d_model: int, d_ff: int | None, dropout: float) -> nn.Module:
    return FeedForward(d_model, d_ff, "gelu", dropout) if kind == "gelu" else GatedFeedForward(d_model, d_ff, "silu")


class TransformerEncoderBlock(nn.Module):
    """Bidirectional self-attention + FFN. forward(x (B, T, d), mask) -> (B, T, d)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.0, norm: str = "layer", pre_norm: bool = True, ffn: str = "gelu") -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = make_ffn(ffn, d_model, d_ff, dropout)
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_norm:
            h = self.norm1(x)  # (B, T, d)
            x = x + self.dropout(self.attn(h, h, mask, bias=bias)[0])  # (B, T, d) residual add
            x = x + self.dropout(self.ffn(self.norm2(x)))  # (B, T, d)
        else:
            x = self.norm1(x + self.dropout(self.attn(x, x, mask, bias=bias)[0]))  # (B, T, d)
            x = self.norm2(x + self.dropout(self.ffn(x)))  # (B, T, d)
        return x


class TransformerDecoderBlock(nn.Module):
    """Causal self-attention, then cross-attention over ``context``, then FFN.

    forward(x (B, T_tgt, d), context (B, T_src, d) or None, self_mask, context_mask, cache) -> (B, T_tgt, d).
    With ``context=None`` this is a decoder-only (GPT) block.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.0, norm: str = "layer", pre_norm: bool = True, ffn: str = "gelu", cross_attention: bool = True) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout) if cross_attention else None
        self.ffn = make_ffn(ffn, d_model, d_ff, dropout)
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model) if cross_attention else None
        self.norm3 = make_norm(norm, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        self_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        cache: LayerKVCache | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        self_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            h = self.norm1(x)  # (B, T_tgt, d)
            x = x + self.dropout(self.self_attn(h, h, self_mask, cache, rope, self_bias)[0])  # (B, T_tgt, d)
            if self.cross_attn is not None and context is not None:
                x = x + self.dropout(self.cross_attn(self.norm2(x), context, context_mask)[0])  # (B, T_tgt, d)
            x = x + self.dropout(self.ffn(self.norm3(x)))  # (B, T_tgt, d)
        else:
            x = self.norm1(x + self.dropout(self.self_attn(x, x, self_mask, cache, rope, self_bias)[0]))  # (B, T_tgt, d)
            if self.cross_attn is not None and context is not None:
                x = self.norm2(x + self.dropout(self.cross_attn(x, context, context_mask)[0]))  # (B, T_tgt, d)
            x = self.norm3(x + self.dropout(self.ffn(x)))  # (B, T_tgt, d)
        return x
