# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/blocks.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k blocks -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/blocks --force

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

    def __init__(self, d_model: int, eps: float=1e-06) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def make_norm(kind: str, d_model: int) -> nn.Module:
    raise NotImplementedError('TODO: implement make_norm (see the reference in src/mlbook)')

def make_ffn(kind: str, d_model: int, d_ff: int | None, dropout: float) -> nn.Module:
    raise NotImplementedError('TODO: implement make_ffn (see the reference in src/mlbook)')

class TransformerEncoderBlock(nn.Module):
    """Bidirectional self-attention + FFN. forward(x (B, T, d), mask) -> (B, T, d)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None=None, dropout: float=0.0, norm: str='layer', pre_norm: bool=True, ffn: str='gelu') -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None=None, bias: torch.Tensor | None=None) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TransformerDecoderBlock(nn.Module):
    """Causal self-attention, then cross-attention over ``context``, then FFN.

    forward(x (B, T_tgt, d), context (B, T_src, d) or None, self_mask, context_mask, cache) -> (B, T_tgt, d).
    With ``context=None`` this is a decoder-only (GPT) block.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None=None, dropout: float=0.0, norm: str='layer', pre_norm: bool=True, ffn: str='gelu', cross_attention: bool=True) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, context: torch.Tensor | None=None, self_mask: torch.Tensor | None=None, context_mask: torch.Tensor | None=None, cache: LayerKVCache | None=None, rope: tuple[torch.Tensor, torch.Tensor] | None=None, self_bias: torch.Tensor | None=None) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
