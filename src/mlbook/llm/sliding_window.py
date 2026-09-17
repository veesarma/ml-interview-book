"""Sparse attention masks: sliding window (Mistral), block-sparse with global
tokens (Longformer/BigBird style), and attention sinks (StreamingLLM), plus a
rolling KV buffer that shows why a window bounds cache memory.

Masks are ``(T, T)`` booleans, ``True`` = query i may attend key j.
"""

from __future__ import annotations

import math

import torch


def sliding_window_mask(T: int, window: int) -> torch.Tensor:
    """Causal mask restricted to the last ``window`` keys (inclusive of self).

    allowed[i, j] = (j <= i) and (i - j < window).   Shape (T, T).
    """
    i = torch.arange(T)[:, None]  # (T, 1)
    j = torch.arange(T)[None, :]  # (1, T)
    return (j <= i) & (i - j < window)  # (T, T)


def attention_sink_mask(T: int, window: int, n_sink: int) -> torch.Tensor:
    """Sliding window plus the first ``n_sink`` tokens always visible (StreamingLLM)."""
    mask = sliding_window_mask(T, window)  # (T, T)
    i = torch.arange(T)[:, None]  # (T, 1)
    j = torch.arange(T)[None, :]  # (1, T)
    return mask | ((j < n_sink) & (j <= i))  # (T, T)


def block_sparse_mask(T: int, block: int, global_tokens: int = 0) -> torch.Tensor:
    """Causal block-local attention (current + previous block) with leading global tokens.

    Tokens attend to every key in their own block and the previous block; the
    first ``global_tokens`` keys are visible to all queries and attend to all keys
    (causally).  Shape (T, T).
    """
    i = torch.arange(T)[:, None]  # (T, 1)
    j = torch.arange(T)[None, :]  # (1, T)
    bi, bj = i // block, j // block  # block index of query / key
    local = (bj == bi) | (bj == bi - 1)  # (T, T)
    glob = (j < global_tokens) | (i < global_tokens)  # (T, T)
    return (local | glob) & (j <= i)  # (T, T)


def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """softmax(QK^T/sqrt(d) + mask) V with a boolean (T, T) mask.

    Args:
        q, k, v: (B, H, T, d_head).
        mask: (T, T) bool.
    Returns:
        (B, H, T, d_head).
    """
    d_head = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)  # (B, H, T, T)
    scores = scores.masked_fill(~mask, float("-inf"))  # (B, H, T, T)
    return torch.softmax(scores, dim=-1) @ v  # (B, H, T, d_head)


class RollingKVCache:
    """Fixed-size ring buffer of the last ``window`` keys/values (Mistral's rolling cache).

    Memory is O(window) regardless of how many tokens have been generated.
    """

    def __init__(self, window: int, n_kv_heads: int, d_head: int) -> None:
        self.window = window
        self.k = torch.zeros(1, n_kv_heads, window, d_head)  # (1, H_kv, W, d_head)
        self.v = torch.zeros(1, n_kv_heads, window, d_head)  # (1, H_kv, W, d_head)
        self.n_seen = 0

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Insert one token's K, V of shape (1, H_kv, 1, d_head) at slot t mod window."""
        slot = self.n_seen % self.window
        self.k[:, :, slot] = k_t[:, :, 0]  # (1, H_kv, d_head)
        self.v[:, :, slot] = v_t[:, :, 0]  # (1, H_kv, d_head)
        self.n_seen += 1

    def window_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached K, V in *chronological* order, shape (1, H_kv, min(n, W), d_head)."""
        n = min(self.n_seen, self.window)
        start = self.n_seen - n
        order = torch.arange(start, self.n_seen) % self.window  # (n,) ring slots, oldest first
        return self.k[:, :, order], self.v[:, :, order]  # (1, H_kv, n, d_head) each
