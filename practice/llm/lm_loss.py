# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/lm_loss.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k lm_loss -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/lm_loss --force

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
    raise NotImplementedError('TODO: implement causal_mask (see the reference in src/mlbook)')

def next_token_loss(logits: torch.Tensor, tokens: torch.Tensor, ignore_index: int=-100) -> torch.Tensor:
    """Mean per-token negative log-likelihood with the causal shift.

    Args:
        logits: (B, T, V) unnormalised scores from a causal LM.
        tokens: (B, T) input token ids (the same tensor that was fed in).
        ignore_index: target id to skip (padding); it contributes neither to the
            sum nor to the token count.
    Returns:
        scalar loss in nats, averaged over non-ignored target positions.
    """
    raise NotImplementedError('TODO: implement next_token_loss (see the reference in src/mlbook)')

def per_token_nll(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Un-averaged NLL for each target position, shape ``(B, T-1)``."""
    raise NotImplementedError('TODO: implement per_token_nll (see the reference in src/mlbook)')

def bits_per_byte(nll_nats_sum: float, n_bytes: int) -> float:
    """Tokeniser-independent loss: total NLL in bits divided by UTF-8 byte count.

    ``BPB = (Σ_t NLL_t / ln 2) / n_bytes``.  Reporting BPB lets you compare
    models with different vocabularies; per-token loss alone does not.
    """
    raise NotImplementedError('TODO: implement bits_per_byte (see the reference in src/mlbook)')

def perplexity(mean_nll_nats: float) -> float:
    """exp of the mean per-token NLL."""
    raise NotImplementedError('TODO: implement perplexity (see the reference in src/mlbook)')
