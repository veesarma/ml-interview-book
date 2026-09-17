# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/attention.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k attention -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/attention --force

"""Scaled dot-product attention: PyTorch forward, and NumPy forward + backward.

    S = Q K^T / sqrt(d_k)        (T_q, T_k)   scores
    A = softmax_rows(S)          (T_q, T_k)   each row sums to 1
    Y = A V                      (T_q, d_v)

Backward (per head; derived in chapter 03):
    dV = A^T dY                              (T_k, d_v)
    dA = dY V^T                              (T_q, T_k)
    dS = A * (dA - rowsum(dA * A))           (T_q, T_k)   softmax Jacobian applied row-wise
    dQ = dS K / sqrt(d_k)                    (T_q, d_k)
    dK = dS^T Q / sqrt(d_k)                  (T_k, d_k)
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F
from .masks import apply_mask

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None=None, dropout_p: float=0.0, training: bool=False, bias: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention for already-projected, already-split heads.

    Args:
        q: (B, H, T_q, d_head) queries.
        k: (B, H, T_k, d_head) keys.
        v: (B, H, T_k, d_head) values.
        mask: bool broadcastable to (B, H, T_q, T_k), True = attend.
        dropout_p: dropout on the attention weights (only if ``training``).
        bias: optional additive score bias broadcastable to (B, H, T_q, T_k)
            (T5 relative-position bias, ALiBi) added *before* masking and softmax.
    Returns:
        out: (B, H, T_q, d_head), attn: (B, H, T_q, T_k) post-softmax weights.
    """
    raise NotImplementedError('TODO: implement scaled_dot_product_attention (see the reference in src/mlbook)')

def softmax_rows(S: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max-subtraction. S: (T_q, T_k) -> (T_q, T_k)."""
    raise NotImplementedError('TODO: implement softmax_rows (see the reference in src/mlbook)')

def attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None=None) -> tuple[np.ndarray, dict]:
    """Single-head attention forward.

    Args:
        Q: (T_q, d_k), K: (T_k, d_k), V: (T_k, d_v); mask: (T_q, T_k) bool, True = attend.
    Returns:
        Y: (T_q, d_v) and a cache for ``attention_backward``.
    """
    raise NotImplementedError('TODO: implement attention_forward (see the reference in src/mlbook)')

def attention_backward(dY: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-head attention backward.

    Args:
        dY: (T_q, d_v) upstream gradient.
    Returns:
        dQ: (T_q, d_k), dK: (T_k, d_k), dV: (T_k, d_v).
    """
    raise NotImplementedError('TODO: implement attention_backward (see the reference in src/mlbook)')

def attention_flops(B: int, T: int, d_model: int) -> dict[str, int]:
    """Forward FLOPs of one multi-head attention layer (all heads together).

    2*B*T*d^2 per projection (Q, K, V, O) and 2*B*T^2*d each for QK^T and AV.
    (A matmul of (m, k) @ (k, n) costs 2*m*k*n FLOPs: one multiply and one add.)
    """
    raise NotImplementedError('TODO: implement attention_flops (see the reference in src/mlbook)')
