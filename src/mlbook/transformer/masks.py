"""Attention masks. Convention used throughout the book: a mask is a boolean tensor
that is **True where attention is allowed** and broadcastable to the score tensor
(B, H, T_q, T_k). Masked positions receive the most negative finite value of the
score dtype *before* the softmax, so they get exactly zero weight after it.

Why not -inf? A fully masked row (a padding *query* attending to nothing) turns
-inf into NaN through softmax(-inf - (-inf)); ``finfo.min`` gives a uniform row
instead, which is harmless because padded outputs are discarded. In fp16 the
"large negative number" must be representable: -1e9 overflows to -inf, -1e4 is
what old BERT code used, ``torch.finfo(dtype).min`` is the portable choice.
"""

from __future__ import annotations

import torch


def causal_mask(T: int, device: torch.device | None = None) -> torch.Tensor:
    """Lower-triangular mask: query i may attend key j iff j <= i.

    Returns:
        (1, 1, T, T) bool, broadcastable over batch and heads.
    """
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))  # (T, T)
    return allowed[None, None, :, :]  # (1, 1, T, T)


def causal_mask_with_cache(T_new: int, T_total: int, device: torch.device | None = None) -> torch.Tensor:
    """Causal mask for the last ``T_new`` query positions against ``T_total`` keys.

    During cached decoding the queries are positions T_total-T_new .. T_total-1 and the
    keys are 0 .. T_total-1. Query at absolute position i may see key j iff j <= i.

    Returns:
        (1, 1, T_new, T_total) bool.
    """
    q_pos = torch.arange(T_total - T_new, T_total, device=device)[:, None]  # (T_new, 1)
    k_pos = torch.arange(T_total, device=device)[None, :]  # (1, T_total)
    return (k_pos <= q_pos)[None, None, :, :]  # (1, 1, T_new, T_total)


def padding_mask(is_pad: torch.Tensor) -> torch.Tensor:
    """Key-padding mask from a (B, T_k) bool tensor that is True at PAD tokens.

    Returns:
        (B, 1, 1, T_k) bool, True where the key is a real token (attention allowed).
    """
    return (~is_pad)[:, None, None, :]  # (B, 1, 1, T_k)


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Logical AND of broadcastable masks (None entries are ignored)."""
    out: torch.Tensor | None = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out & m)
    return out


def apply_mask(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Fill disallowed positions with the most negative finite value of the dtype.

    scores: (B, H, T_q, T_k) pre-softmax; mask broadcastable bool (True = keep).
    """
    if mask is None:
        return scores
    return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # (B, H, T_q, T_k)
