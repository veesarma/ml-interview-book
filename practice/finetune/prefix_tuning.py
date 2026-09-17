# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/finetune/prefix_tuning.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k prefix_tuning -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py finetune/prefix_tuning --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, n_virtual + T, d_model)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class PrefixKV(nn.Module):
    """Per-layer learned prefix keys and values for one attention layer."""

    def __init__(self, n_prefix: int, n_kv_heads: int, d_head: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def expand(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast to a batch: (B, H_kv, P, d_head) each."""
        raise NotImplementedError('TODO: implement expand (see the reference in src/mlbook)')

def attention_with_prefix(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, prefix_k: torch.Tensor, prefix_v: torch.Tensor) -> torch.Tensor:
    """Causal attention where every query also sees the ``P`` prefix positions.

    Args:
        q, k, v:            (B, H, T, d_head)  (K/V already expanded to H heads).
        prefix_k, prefix_v: (B, H, P, d_head).
    Returns:
        (B, H, T, d_head).
    """
    raise NotImplementedError('TODO: implement attention_with_prefix (see the reference in src/mlbook)')
