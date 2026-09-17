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
    m = torch.tensor(float("-inf"))  # running max
    l = torch.tensor(0.0)  # running sum of exp(x - m)
    for start in range(0, x.shape[0], chunk):
        blk = x[start : start + chunk]  # (chunk,)
        m_new = torch.maximum(m, blk.max())
        l = l * torch.exp(m - m_new) + torch.exp(blk - m_new).sum()
        m = m_new
    return torch.exp(x - m) / l  # (T,)


def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """Reference: materialise the full (T, T) score matrix.  q, k, v: (B, H, T, d_head)."""
    T = q.shape[-2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, T)
    if causal:
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # (T, T)
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v  # (B, H, T, d_head)


def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_q: int, block_kv: int, causal: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise exact attention; never allocates a (T, T) matrix.

    Args:
        q, k, v: (B, H, T, d_head).
        block_q, block_kv: tile sizes (in a kernel these are set so tiles fit SRAM).
    Returns:
        out: (B, H, T, d_head) = softmax(QK^T/√d) V.
        lse: (B, H, T) row log-sum-exp, saved for the backward pass (recomputation).
    """
    B, H, T, d_head = q.shape
    scale = 1.0 / math.sqrt(d_head)
    out = torch.zeros_like(q)  # (B, H, T, d_head)
    lse = torch.zeros(B, H, T)  # (B, H, T)
    for qs in range(0, T, block_q):
        q_blk = q[:, :, qs : qs + block_q]  # (B, H, bq, d_head)
        bq = q_blk.shape[2]
        m = torch.full((B, H, bq), float("-inf"))  # (B, H, bq) running row max
        l = torch.zeros(B, H, bq)  # (B, H, bq) running row sum
        acc = torch.zeros(B, H, bq, d_head)  # (B, H, bq, d_head) un-normalised output
        kv_end = min(T, qs + bq) if causal else T  # causal: skip fully-masked K/V blocks
        for ks in range(0, kv_end, block_kv):
            k_blk = k[:, :, ks : ks + block_kv]  # (B, H, bk, d_head)
            v_blk = v[:, :, ks : ks + block_kv]  # (B, H, bk, d_head)
            s = q_blk @ k_blk.transpose(-2, -1) * scale  # (B, H, bq, bk)
            if causal:
                qi = torch.arange(qs, qs + bq)[:, None]  # (bq, 1) absolute query positions
                kj = torch.arange(ks, ks + k_blk.shape[2])[None, :]  # (1, bk)
                s = s.masked_fill(kj > qi, float("-inf"))  # (B, H, bq, bk)
            m_new = torch.maximum(m, s.amax(dim=-1))  # (B, H, bq)
            p = torch.exp(s - m_new[..., None])  # (B, H, bq, bk) exp against the *new* max
            alpha = torch.exp(m - m_new)  # (B, H, bq) rescale for the old max
            l = alpha * l + p.sum(dim=-1)  # (B, H, bq)
            acc = alpha[..., None] * acc + p @ v_blk  # (B, H, bq, d_head)
            m = m_new
        out[:, :, qs : qs + bq] = acc / l[..., None]  # (B, H, bq, d_head)
        lse[:, :, qs : qs + bq] = m + torch.log(l)  # (B, H, bq)
    return out, lse


def attention_hbm_bytes_standard(T: int, d_head: int, dtype_bytes: int = 2) -> int:
    """HBM traffic of naive attention per head: read Q,K,V; write S; read S; write P; read P; write O.

    S and P are T×T, so traffic is Θ(T² ) bytes; Q, K, V, O are Θ(T d).
    """
    qkvo = 4 * T * d_head * dtype_bytes
    s_and_p = 4 * T * T * dtype_bytes  # write S, read S, write P, read P
    return qkvo + s_and_p


def attention_hbm_bytes_flash(T: int, d_head: int, sram_bytes: int, dtype_bytes: int = 2) -> float:
    """FlashAttention HBM traffic per head: Θ(T² d² / M) with M = SRAM size (bytes).

    Each pass over K, V (T d bytes) happens once per query block; the number of query
    blocks is T d / (M / dtype_bytes) up to constants, giving T·d · T·d / M reads.
    """
    kv_bytes = 2 * T * d_head * dtype_bytes
    n_query_blocks = max(1.0, (T * d_head * dtype_bytes) / sram_bytes)
    return 2 * T * d_head * dtype_bytes + n_query_blocks * kv_bytes
