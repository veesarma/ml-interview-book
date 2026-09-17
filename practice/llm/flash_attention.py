# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/flash_attention.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k flash_attention -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/flash_attention --force

"""An educational FlashAttention forward pass: blockwise attention with online softmax.

Standard attention materialises S = QK^T (T×T) and P = softmax(S) in HBM.
FlashAttention never does: for each block of queries it streams blocks of K, V
through SRAM, keeping a running max ``m``, running denominator ``l`` and
un-normalised accumulator ``acc``:

    m_new = max(m, rowmax(S_blk))
    l_new = exp(m − m_new) · l + rowsum(exp(S_blk − m_new))
    acc   = exp(m − m_new) · acc + exp(S_blk − m_new) @ V_blk
    O     = acc / l  at the end.

The rescale factor exp(m − m_new) corrects everything accumulated under the old
max, so the result is *exactly* softmax(S)V, just computed in a different order.
"""
from __future__ import annotations
import math
import torch

def online_softmax(x: torch.Tensor, chunk: int) -> torch.Tensor:
    """softmax over a 1-D tensor computed in one streaming pass over ``chunk``-sized pieces.

    Args:
        x: (T,).
    Returns:
        (T,) identical (up to float error) to torch.softmax(x, 0).
    """
    raise NotImplementedError('TODO: implement online_softmax (see the reference in src/mlbook)')

def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """Reference: materialise the full (T, T) score matrix.  q, k, v: (B, H, T, d_head)."""
    raise NotImplementedError('TODO: implement standard_attention (see the reference in src/mlbook)')

def flash_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_q: int, block_kv: int, causal: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise exact attention; never allocates a (T, T) matrix.

    Args:
        q, k, v: (B, H, T, d_head).
        block_q, block_kv: tile sizes (in a kernel these are set so tiles fit SRAM).
    Returns:
        out: (B, H, T, d_head) = softmax(QK^T/√d) V.
        lse: (B, H, T) row log-sum-exp, saved for the backward pass (recomputation).
    """
    raise NotImplementedError('TODO: implement flash_attention_forward (see the reference in src/mlbook)')

def attention_hbm_bytes_standard(T: int, d_head: int, dtype_bytes: int=2) -> int:
    """HBM traffic of naive attention per head: read Q,K,V; write S; read S; write P; read P; write O.

    S and P are T×T, so traffic is Θ(T² ) bytes; Q, K, V, O are Θ(T d).
    """
    raise NotImplementedError('TODO: implement attention_hbm_bytes_standard (see the reference in src/mlbook)')

def attention_hbm_bytes_flash(T: int, d_head: int, sram_bytes: int, dtype_bytes: int=2) -> float:
    """FlashAttention HBM traffic per head: Θ(T² d² / M) with M = SRAM size (bytes).

    Each pass over K, V (T d bytes) happens once per query block; the number of query
    blocks is T d / (M / dtype_bytes) up to constants, giving T·d · T·d / M reads.
    """
    raise NotImplementedError('TODO: implement attention_hbm_bytes_flash (see the reference in src/mlbook)')
