# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/positional.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k positional -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/positional --force

"""Positional encodings: sinusoidal, learned, T5 relative bias, RoPE, ALiBi.

Attention is permutation-equivariant -- softmax(QK^T)V does not know where a token
sits -- so position must be injected. Four places to do it:

    add to the input:    sinusoidal (fixed) or learned absolute embeddings
    add to the scores:   T5 relative bias b[bucket(i - j)], ALiBi  -m * (i - j)
    rotate q and k:      RoPE, so q_m . k_n depends only on m - n
"""
from __future__ import annotations
import math
import torch
from torch import nn

def sinusoidal_positional_encoding(T: int, d_model: int, base: float=10000.0) -> torch.Tensor:
    """PE[t, 2i] = sin(t / base^(2i/d)), PE[t, 2i+1] = cos(t / base^(2i/d)).

    Returns:
        (T, d_model) float tensor; row t is the encoding of position t.
    """
    raise NotImplementedError('TODO: implement sinusoidal_positional_encoding (see the reference in src/mlbook)')

class LearnedPositionalEmbedding(nn.Module):
    """A (T_max, d_model) table indexed by position (GPT-2, BERT). Cannot extrapolate past T_max."""

    def __init__(self, T_max: int, d_model: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, T: int, offset: int=0) -> torch.Tensor:
        """Returns (1, T, d_model) for positions offset .. offset+T-1 (offset > 0 during cached decode)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool, num_buckets: int, max_distance: int) -> torch.Tensor:
    """T5's bucketing: exact buckets for small |i - j|, log-spaced buckets up to max_distance.

    relative_position: (T_q, T_k) int tensor = key_pos - query_pos. Returns bucket ids of the same shape.
    """
    raise NotImplementedError('TODO: implement relative_position_bucket (see the reference in src/mlbook)')

class RelativePositionBias(nn.Module):
    """Learned scalar per (head, bucket), added to attention scores. Shared across layers in T5."""

    def __init__(self, n_heads: int, num_buckets: int=32, max_distance: int=128, bidirectional: bool=True) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, T_q: int, T_k: int) -> torch.Tensor:
        """Returns (1, H, T_q, T_k) bias."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021) with optional context-extension scaling.

    Pair the d_head dimensions as (x_i, x_{i + d/2}) for i < d/2 (the "rotate-half" layout
    used by GPT-NeoX / LLaMA) and rotate pair i at position m by angle m * theta_i with
        theta_i = base^(-2i / d_head).
    Then <R_m q, R_n k> = <q, R_{n-m} k>: the score depends only on the offset n - m.

    scaling="linear" (position interpolation): use position m / factor.
    scaling="ntk":   keep positions, enlarge the base to base * factor^(d/(d-2)) so the
                     lowest frequency stretches by ~factor while high frequencies barely move.
    """

    def __init__(self, d_head: int, base: float=10000.0, scaling: str='none', factor: float=1.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, T: int, offset: int=0) -> tuple[torch.Tensor, torch.Tensor]:
        """cos, sin tables for positions offset .. offset+T-1, each (T, d_head)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., d) -> (..., d): [x1, x2] -> [-x2, x1] where x1, x2 are the two halves."""
    raise NotImplementedError('TODO: implement rotate_half (see the reference in src/mlbook)')

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (B, H, T, d_head) by per-position angles. cos, sin: (T, d_head).

    Per pair (a, b) at angle t:  (a cos t - b sin t, b cos t + a sin t), i.e. multiplication
    by e^{i t} if you read the pair as the complex number a + ib.
    """
    raise NotImplementedError('TODO: implement apply_rotary (see the reference in src/mlbook)')

def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric slopes 2^(-8/H), 2^(-16/H), ... (Press et al., 2022) for H a power of two.

    Returns (H,) tensor.
    """
    raise NotImplementedError('TODO: implement alibi_slopes (see the reference in src/mlbook)')

def alibi_bias(n_heads: int, T_q: int, T_k: int) -> torch.Tensor:
    """Causal ALiBi bias: score[h, i, j] += -m_h * (i - j) for j <= i (masking handles j > i).

    Returns (1, H, T_q, T_k). Queries are assumed to be the last T_q of the T_k positions.
    """
    raise NotImplementedError('TODO: implement alibi_bias (see the reference in src/mlbook)')
