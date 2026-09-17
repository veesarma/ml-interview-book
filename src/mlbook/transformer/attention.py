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

# ---------------------------------------------------------------------------
# PyTorch (batched, multi-head)
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    d_head = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)  # (B, H, T_q, d_head) @ (B, H, d_head, T_k) -> (B, H, T_q, T_k)
    if bias is not None:
        scores = scores + bias  # (B, H, T_q, T_k) relative-position / ALiBi bias
    scores = apply_mask(scores, mask)  # (B, H, T_q, T_k)
    attn = F.softmax(scores, dim=-1)  # (B, H, T_q, T_k) rows sum to 1
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)  # (B, H, T_q, T_k)
    out = attn @ v  # (B, H, T_q, T_k) @ (B, H, T_k, d_head) -> (B, H, T_q, d_head)
    return out, attn


# ---------------------------------------------------------------------------
# NumPy (single head, single sequence) -- forward and backward by hand
# ---------------------------------------------------------------------------


def softmax_rows(S: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max-subtraction. S: (T_q, T_k) -> (T_q, T_k)."""
    S = S - S.max(axis=-1, keepdims=True)  # (T_q, T_k) shift for stability
    E = np.exp(S)  # (T_q, T_k)
    return E / E.sum(axis=-1, keepdims=True)  # (T_q, T_k)


def attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Single-head attention forward.

    Args:
        Q: (T_q, d_k), K: (T_k, d_k), V: (T_k, d_v); mask: (T_q, T_k) bool, True = attend.
    Returns:
        Y: (T_q, d_v) and a cache for ``attention_backward``.
    """
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)  # (T_q, d_k) @ (d_k, T_k) -> (T_q, T_k)
    if mask is not None:
        S = np.where(mask, S, -1e30)  # (T_q, T_k) masked scores get zero weight
    A = softmax_rows(S)  # (T_q, T_k)
    Y = A @ V  # (T_q, T_k) @ (T_k, d_v) -> (T_q, d_v)
    return Y, {"Q": Q, "K": K, "V": V, "A": A, "d_k": d_k}


def attention_backward(dY: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-head attention backward.

    Args:
        dY: (T_q, d_v) upstream gradient.
    Returns:
        dQ: (T_q, d_k), dK: (T_k, d_k), dV: (T_k, d_v).
    """
    Q, K, V, A, d_k = cache["Q"], cache["K"], cache["V"], cache["A"], cache["d_k"]
    dV = A.T @ dY  # (T_k, T_q) @ (T_q, d_v) -> (T_k, d_v)
    dA = dY @ V.T  # (T_q, d_v) @ (d_v, T_k) -> (T_q, T_k)
    # softmax backward, row by row: dS_i = A_i * (dA_i - <dA_i, A_i>)
    dS = A * (dA - (dA * A).sum(axis=-1, keepdims=True))  # (T_q, T_k)
    dS = dS / np.sqrt(d_k)  # (T_q, T_k) undo the scale
    dQ = dS @ K  # (T_q, T_k) @ (T_k, d_k) -> (T_q, d_k)
    dK = dS.T @ Q  # (T_k, T_q) @ (T_q, d_k) -> (T_k, d_k)
    return dQ, dK, dV


def attention_flops(B: int, T: int, d_model: int) -> dict[str, int]:
    """Forward FLOPs of one multi-head attention layer (all heads together).

    2*B*T*d^2 per projection (Q, K, V, O) and 2*B*T^2*d each for QK^T and AV.
    (A matmul of (m, k) @ (k, n) costs 2*m*k*n FLOPs: one multiply and one add.)
    """
    proj = 4 * 2 * B * T * d_model * d_model
    scores = 2 * B * T * T * d_model
    weighted_sum = 2 * B * T * T * d_model
    return {"projections": proj, "qk_t": scores, "av": weighted_sum, "total": proj + scores + weighted_sum}
