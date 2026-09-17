"""Prefix tuning and prompt tuning: steer a frozen model with learned virtual tokens.

Prompt tuning  (Lester et al. 2021): prepend ``n_virtual`` learned embeddings to the
input embeddings; everything else is frozen.  Parameters: n_virtual · d_model.

Prefix tuning  (Li & Liang 2021): prepend learned key/value vectors at *every* layer's
attention, so each real token attends to ``n_prefix`` extra positions:

    softmax( q [P_K ; K]^T / √d ) [P_V ; V]

Parameters: L · 2 · n_prefix · H_kv · d_head.  No extra tokens enter the residual
stream, and the prefix behaves like a cached prompt that was never actually written.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PromptTuning(nn.Module):
    """Learned virtual-token embeddings prepended to the token embeddings."""

    def __init__(self, n_virtual: int, d_model: int) -> None:
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(n_virtual, d_model) * 0.02)  # (n_virtual, d_model)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, n_virtual + T, d_model)."""
        B = token_embeddings.shape[0]
        prompt = self.prompt[None].expand(B, -1, -1)  # (B, n_virtual, d_model)
        return torch.cat([prompt, token_embeddings], dim=1)  # (B, n_virtual + T, d_model)


class PrefixKV(nn.Module):
    """Per-layer learned prefix keys and values for one attention layer."""

    def __init__(self, n_prefix: int, n_kv_heads: int, d_head: int) -> None:
        super().__init__()
        self.prefix_k = nn.Parameter(torch.randn(n_kv_heads, n_prefix, d_head) * 0.02)  # (H_kv, P, d_head)
        self.prefix_v = nn.Parameter(torch.randn(n_kv_heads, n_prefix, d_head) * 0.02)  # (H_kv, P, d_head)

    def expand(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast to a batch: (B, H_kv, P, d_head) each."""
        return self.prefix_k[None].expand(B, -1, -1, -1), self.prefix_v[None].expand(B, -1, -1, -1)


def attention_with_prefix(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, prefix_k: torch.Tensor, prefix_v: torch.Tensor
) -> torch.Tensor:
    """Causal attention where every query also sees the ``P`` prefix positions.

    Args:
        q, k, v:            (B, H, T, d_head)  (K/V already expanded to H heads).
        prefix_k, prefix_v: (B, H, P, d_head).
    Returns:
        (B, H, T, d_head).
    """
    B, H, T, d_head = q.shape
    P = prefix_k.shape[2]
    k_all = torch.cat([prefix_k, k], dim=2)  # (B, H, P+T, d_head)
    v_all = torch.cat([prefix_v, v], dim=2)  # (B, H, P+T, d_head)
    scores = q @ k_all.transpose(-2, -1) / math.sqrt(d_head)  # (B, H, T, P+T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))  # (T, T) for the real keys
    allowed = torch.cat([torch.ones(T, P, dtype=torch.bool), causal], dim=1)  # (T, P+T)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v_all  # (B, H, T, d_head)
