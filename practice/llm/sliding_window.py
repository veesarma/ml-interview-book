# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/sliding_window.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k sliding_window -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/sliding_window --force

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
    raise NotImplementedError('TODO: implement sliding_window_mask (see the reference in src/mlbook)')

def attention_sink_mask(T: int, window: int, n_sink: int) -> torch.Tensor:
    """Sliding window plus the first ``n_sink`` tokens always visible (StreamingLLM)."""
    raise NotImplementedError('TODO: implement attention_sink_mask (see the reference in src/mlbook)')

def block_sparse_mask(T: int, block: int, global_tokens: int=0) -> torch.Tensor:
    """Causal block-local attention (current + previous block) with leading global tokens.

    Tokens attend to every key in their own block and the previous block; the
    first ``global_tokens`` keys are visible to all queries and attend to all keys
    (causally).  Shape (T, T).
    """
    raise NotImplementedError('TODO: implement block_sparse_mask (see the reference in src/mlbook)')

def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """softmax(QK^T/sqrt(d) + mask) V with a boolean (T, T) mask.

    Args:
        q, k, v: (B, H, T, d_head).
        mask: (T, T) bool.
    Returns:
        (B, H, T, d_head).
    """
    raise NotImplementedError('TODO: implement masked_attention (see the reference in src/mlbook)')

class RollingKVCache:
    """Fixed-size ring buffer of the last ``window`` keys/values (Mistral's rolling cache).

    Memory is O(window) regardless of how many tokens have been generated.
    """

    def __init__(self, window: int, n_kv_heads: int, d_head: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Insert one token's K, V of shape (1, H_kv, 1, d_head) at slot t mod window."""
        raise NotImplementedError('TODO: implement append (see the reference in src/mlbook)')

    def window_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached K, V in *chronological* order, shape (1, H_kv, min(n, W), d_head)."""
        raise NotImplementedError('TODO: implement window_kv (see the reference in src/mlbook)')
