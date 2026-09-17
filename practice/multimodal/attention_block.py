# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/attention_block.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k attention_block -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/attention_block --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, N, d) -> (B, H, N, d_head)."""
        raise NotImplementedError('TODO: implement split_heads (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: queries from ``x`` (B, N_q, d), keys/values from ``ctx`` (B, N_kv, d_ctx).

    Output (B, N_q, d). Used by DETR decoders, Perceiver resamplers and
    Flamingo-style gated adapters.
    """

    def __init__(self, d: int, d_ctx: int, n_heads: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class MLP(nn.Module):
    """Position-wise MLP: (B, N, d) -> (B, N, d) through a ``mlp_ratio * d`` hidden layer with GELU."""

    def __init__(self, d: int, mlp_ratio: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)); then x + MLP(LN(x)). (B, N, d) -> (B, N, d)."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
