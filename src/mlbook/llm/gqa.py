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
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.d_head = d_model // n_heads
        self.group = n_heads // n_kv_heads  # query heads per KV head
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, T, d_model) -> q (B, H, T, d_head), k/v (B, H_kv, T, d_head)."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # (B, H_kv, T, d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # (B, H_kv, T, d_head)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal self-attention. (B, T, d_model) -> (B, T, d_model)."""
        B, T, _ = x.shape
        q, k, v = self.project(x)
        # Expand each KV head to its `group` query heads: head h reads kv head h // group.
        k = k.repeat_interleave(self.group, dim=1)  # (B, H, T, d_head)
        v = v.repeat_interleave(self.group, dim=1)  # (B, H, T, d_head)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))  # (T, T)
        scores = scores.masked_fill(~causal, float("-inf"))  # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        out = attn @ v  # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)  # (B, T, H*d_head)
        return self.o_proj(out)  # (B, T, d_model)

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """KV-cache bytes per token for *this layer*: 2 · H_kv · d_head · bytes."""
        return 2 * self.n_kv_heads * self.d_head * dtype_bytes


def kv_cache_reduction_factor(n_heads: int, n_kv_heads: int) -> float:
    """H / H_kv: how much smaller the KV cache is than MHA's."""
    return n_heads / n_kv_heads
