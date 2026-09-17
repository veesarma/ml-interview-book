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

# ---------------------------------------------------------------------------
# Absolute: sinusoidal and learned
# ---------------------------------------------------------------------------


def sinusoidal_positional_encoding(T: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """PE[t, 2i] = sin(t / base^(2i/d)), PE[t, 2i+1] = cos(t / base^(2i/d)).

    Returns:
        (T, d_model) float tensor; row t is the encoding of position t.
    """
    pos = torch.arange(T, dtype=torch.float32)[:, None]  # (T, 1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)  # (d_model/2,) the pair index 2i
    inv_freq = base ** (-i / d_model)  # (d_model/2,) omega_i = base^(-2i/d)
    angles = pos * inv_freq  # (T, d_model/2) t * omega_i
    pe = torch.zeros(T, d_model)  # (T, d_model)
    pe[:, 0::2] = torch.sin(angles)  # even columns
    pe[:, 1::2] = torch.cos(angles)  # odd columns
    return pe


class LearnedPositionalEmbedding(nn.Module):
    """A (T_max, d_model) table indexed by position (GPT-2, BERT). Cannot extrapolate past T_max."""

    def __init__(self, T_max: int, d_model: int) -> None:
        super().__init__()
        self.table = nn.Embedding(T_max, d_model)  # (T_max, d_model)

    def forward(self, T: int, offset: int = 0) -> torch.Tensor:
        """Returns (1, T, d_model) for positions offset .. offset+T-1 (offset > 0 during cached decode)."""
        pos = torch.arange(offset, offset + T, device=self.table.weight.device)  # (T,)
        return self.table(pos)[None, :, :]  # (1, T, d_model)


# ---------------------------------------------------------------------------
# Relative: T5 bucketed bias
# ---------------------------------------------------------------------------


def relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool, num_buckets: int, max_distance: int) -> torch.Tensor:
    """T5's bucketing: exact buckets for small |i - j|, log-spaced buckets up to max_distance.

    relative_position: (T_q, T_k) int tensor = key_pos - query_pos. Returns bucket ids of the same shape.
    """
    ret = torch.zeros_like(relative_position)
    n = -relative_position  # (T_q, T_k) positive when the key is in the past
    if bidirectional:
        num_buckets //= 2
        ret = ret + (n < 0).long() * num_buckets  # future keys use the upper half of the buckets
        n = n.abs()
    else:
        n = torch.clamp(n, min=0)  # causal: future keys collapse into bucket 0 (they are masked anyway)
    max_exact = num_buckets // 2
    is_small = n < max_exact  # (T_q, T_k) these distances get their own bucket
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact + 1e-6) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).long()  # (T_q, T_k) log-spaced
    val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
    return ret + torch.where(is_small, n, val_if_large)  # (T_q, T_k)


class RelativePositionBias(nn.Module):
    """Learned scalar per (head, bucket), added to attention scores. Shared across layers in T5."""

    def __init__(self, n_heads: int, num_buckets: int = 32, max_distance: int = 128, bidirectional: bool = True) -> None:
        super().__init__()
        self.num_buckets, self.max_distance, self.bidirectional = num_buckets, max_distance, bidirectional
        self.table = nn.Embedding(num_buckets, n_heads)  # (num_buckets, H)

    def forward(self, T_q: int, T_k: int) -> torch.Tensor:
        """Returns (1, H, T_q, T_k) bias."""
        device = self.table.weight.device
        q_pos = torch.arange(T_q, device=device)[:, None]  # (T_q, 1)
        k_pos = torch.arange(T_k, device=device)[None, :]  # (1, T_k)
        buckets = relative_position_bucket(k_pos - q_pos, self.bidirectional, self.num_buckets, self.max_distance)  # (T_q, T_k)
        bias = self.table(buckets)  # (T_q, T_k, H)
        return bias.permute(2, 0, 1)[None, :, :, :]  # (1, H, T_q, T_k)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


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

    def __init__(self, d_head: int, base: float = 10000.0, scaling: str = "none", factor: float = 1.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError("d_head must be even")
        self.d_head, self.scaling, self.factor = d_head, scaling, factor
        if scaling == "ntk":
            base = base * factor ** (d_head / (d_head - 2))
        i = torch.arange(0, d_head, 2, dtype=torch.float32)  # (d_head/2,) = 2i
        inv_freq = base ** (-i / d_head)  # (d_head/2,) theta_i
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, T: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """cos, sin tables for positions offset .. offset+T-1, each (T, d_head)."""
        pos = torch.arange(offset, offset + T, device=self.inv_freq.device, dtype=torch.float32)  # (T,)
        if self.scaling == "linear":
            pos = pos / self.factor  # position interpolation squeezes positions into the trained range
        angles = pos[:, None] * self.inv_freq[None, :]  # (T, d_head/2) m * theta_i
        angles = torch.cat([angles, angles], dim=-1)  # (T, d_head) same angle for both halves of a pair
        return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., d) -> (..., d): [x1, x2] -> [-x2, x1] where x1, x2 are the two halves."""
    x1 = x[..., : x.shape[-1] // 2]  # (..., d/2)
    x2 = x[..., x.shape[-1] // 2 :]  # (..., d/2)
    return torch.cat([-x2, x1], dim=-1)  # (..., d)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (B, H, T, d_head) by per-position angles. cos, sin: (T, d_head).

    Per pair (a, b) at angle t:  (a cos t - b sin t, b cos t + a sin t), i.e. multiplication
    by e^{i t} if you read the pair as the complex number a + ib.
    """
    return x * cos[None, None, :, :] + rotate_half(x) * sin[None, None, :, :]  # (B, H, T, d_head)


# ---------------------------------------------------------------------------
# ALiBi
# ---------------------------------------------------------------------------


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric slopes 2^(-8/H), 2^(-16/H), ... (Press et al., 2022) for H a power of two.

    Returns (H,) tensor.
    """
    if n_heads & (n_heads - 1) != 0:
        raise ValueError("this helper assumes n_heads is a power of two")
    start = 2.0 ** (-(2.0 ** -(math.log2(n_heads) - 3)))  # 2^(-8/H)
    return torch.tensor([start ** (i + 1) for i in range(n_heads)])  # (H,)


def alibi_bias(n_heads: int, T_q: int, T_k: int) -> torch.Tensor:
    """Causal ALiBi bias: score[h, i, j] += -m_h * (i - j) for j <= i (masking handles j > i).

    Returns (1, H, T_q, T_k). Queries are assumed to be the last T_q of the T_k positions.
    """
    q_pos = torch.arange(T_k - T_q, T_k)[:, None]  # (T_q, 1)
    k_pos = torch.arange(T_k)[None, :]  # (1, T_k)
    distance = (q_pos - k_pos).clamp(min=0).float()  # (T_q, T_k) i - j, zero for the future
    slopes = alibi_slopes(n_heads)  # (H,)
    return -slopes[None, :, None, None] * distance[None, None, :, :]  # (1, H, T_q, T_k)
