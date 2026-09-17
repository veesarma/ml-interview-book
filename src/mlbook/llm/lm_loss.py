"""The next-token prediction objective, written out.

    L(θ) = - (1 / N_targets) Σ_{b,t} log p_θ(x_{b,t+1} | x_{b,≤t})

Teacher forcing: a single causal forward pass produces logits at every
position, and position t is scored against the *true* token t+1, so one
forward pass yields T-1 losses per row.  The shift is the whole trick:
``logits[:, :-1]`` predicts ``tokens[:, 1:]``.
"""

from __future__ import annotations

import math

import torch


def causal_mask(T: int) -> torch.Tensor:
    """(T, T) boolean lower-triangular mask: query i may attend key j iff j ≤ i."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))  # (T, T)


def next_token_loss(
    logits: torch.Tensor, tokens: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Mean per-token negative log-likelihood with the causal shift.

    Args:
        logits: (B, T, V) unnormalised scores from a causal LM.
        tokens: (B, T) input token ids (the same tensor that was fed in).
        ignore_index: target id to skip (padding); it contributes neither to the
            sum nor to the token count.
    Returns:
        scalar loss in nats, averaged over non-ignored target positions.
    """
    B, T, V = logits.shape
    pred = logits[:, :-1, :]  # (B, T-1, V)  position t predicts token t+1
    target = tokens[:, 1:]  # (B, T-1)
    log_probs = torch.log_softmax(pred.float(), dim=-1)  # (B, T-1, V)
    valid = target != ignore_index  # (B, T-1)
    safe_target = target.masked_fill(~valid, 0)  # (B, T-1)
    picked = log_probs.gather(-1, safe_target[..., None]).squeeze(-1)  # (B, T-1)
    nll = -(picked * valid).sum()  # scalar, sum over valid targets
    return nll / valid.sum().clamp(min=1)


def per_token_nll(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Un-averaged NLL for each target position, shape ``(B, T-1)``."""
    pred = logits[:, :-1, :]  # (B, T-1, V)
    target = tokens[:, 1:]  # (B, T-1)
    log_probs = torch.log_softmax(pred.float(), dim=-1)  # (B, T-1, V)
    return -log_probs.gather(-1, target[..., None]).squeeze(-1)  # (B, T-1)


def bits_per_byte(nll_nats_sum: float, n_bytes: int) -> float:
    """Tokeniser-independent loss: total NLL in bits divided by UTF-8 byte count.

    ``BPB = (Σ_t NLL_t / ln 2) / n_bytes``.  Reporting BPB lets you compare
    models with different vocabularies; per-token loss alone does not.
    """
    return nll_nats_sum / math.log(2) / n_bytes


def perplexity(mean_nll_nats: float) -> float:
    """exp of the mean per-token NLL."""
    return math.exp(mean_nll_nats)
