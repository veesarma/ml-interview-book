# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/gqa.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gqa -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/gqa --force

"""Grouped-query attention (GQA), with MHA and MQA as the two extremes.

``H`` query heads share ``H_kv`` key/value heads; every group of ``H / H_kv``
consecutive query heads reads the same K, V.  The KV cache per token shrinks by
``H / H_kv``:  MHA (H_kv = H) -> factor 1,  MQA (H_kv = 1) -> factor H.

Implementation is explicit: three separate projections, ``repeat_interleave`` to
expand K/V to the query-head count, a causal mask, softmax, output projection.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """Causal GQA.  ``n_kv_heads == n_heads`` is MHA; ``n_kv_heads == 1`` is MQA."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, T, d_model) -> q (B, H, T, d_head), k/v (B, H_kv, T, d_head)."""
        raise NotImplementedError('TODO: implement project (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal self-attention. (B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def kv_bytes_per_token(self, dtype_bytes: int=2) -> int:
        """KV-cache bytes per token for *this layer*: 2 · H_kv · d_head · bytes."""
        raise NotImplementedError('TODO: implement kv_bytes_per_token (see the reference in src/mlbook)')

def kv_cache_reduction_factor(n_heads: int, n_kv_heads: int) -> float:
    """H / H_kv: how much smaller the KV cache is than MHA's."""
    raise NotImplementedError('TODO: implement kv_cache_reduction_factor (see the reference in src/mlbook)')
