# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/multihead.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k multihead -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/multihead --force

"""Multi-head attention with three separate Q/K/V projections (never a fused 3d one).

    head_i = Attention(X W_i^Q, X_kv W_i^K, X_kv W_i^V)     i = 1..H
    MHA(X) = Concat(head_1, ..., head_H) W^O

Implementation trick: one ``nn.Linear(d_model, d_model)`` for Q computes all H heads at
once because W^Q = [W_1^Q | ... | W_H^Q]; splitting the last dimension into
(H, d_head) and moving H next to the batch axis gives the per-head tensors.
"""
from __future__ import annotations
import torch
from torch import nn
from .attention import scaled_dot_product_attention
from .kv_cache import LayerKVCache
from .positional import apply_rotary

def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, d_model) -> (B, H, T, d_head).

    Step 1 view:      (B, T, d_model) -> (B, T, H, d_head)   [no data movement]
    Step 2 transpose: (B, T, H, d_head) -> (B, H, T, d_head) [so matmuls batch over (B, H)]
    """
    raise NotImplementedError('TODO: implement split_heads (see the reference in src/mlbook)')

def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, d_head) -> (B, T, d_model). The concat in Concat(head_1..head_H)."""
    raise NotImplementedError('TODO: implement merge_heads (see the reference in src/mlbook)')

class MultiHeadAttention(nn.Module):
    """Self- or cross-attention depending on what is passed as ``x_kv``.

    forward(x_q, x_kv, mask, cache) -> (out (B, T_q, d_model), attn (B, H, T_q, T_k)).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0, bias: bool=True) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: torch.Tensor | None=None, cache: LayerKVCache | None=None, rope: tuple[torch.Tensor, torch.Tensor] | None=None, bias: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        """x_q: (B, T_q, d_model); x_kv: (B, T_k, d_model). Self-attention when x_kv is x_q.

        With ``cache`` the keys/values of ``x_kv`` are appended to the cache and attention
        runs over the whole cached prefix (T_k becomes cache.length).
        ``rope`` = (cos, sin), each (T_q, d_head) for the *new* positions, rotates q and k
        (self-attention only). ``bias`` is an additive score bias (T5 / ALiBi).
        """
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class CrossAttention(nn.Module):
    """Queries from the decoder stream ``x``; keys and values from ``context`` (encoder output,
    image tokens, retrieved passages...). Output has the query's length, always."""

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T_q, d), context: (B, T_ctx, d), context_mask: (B, 1, 1, T_ctx) -> (B, T_q, d), (B, H, T_q, T_ctx)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
