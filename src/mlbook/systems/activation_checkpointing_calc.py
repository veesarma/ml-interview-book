"""Activation checkpointing: memory vs recompute trade-off for L layers.

Let A = activation bytes one layer must keep for its backward pass, I = bytes of a
layer's input (2 s b h in 16-bit), and F = forward FLOPs of one layer.

    none      : memory L A,                     extra compute 0
    full      : memory L I + A,                 extra compute L F   (+1/3 of a train step)
    sqrt      : c = ceil(sqrt(L)) segments:
                memory c I + (L/c) A,           extra compute L F   (Chen et al. 2016)
    selective : keep everything except the attention-score tensors (Korthikanti 2022):
                memory L A_sel,                 extra compute = the attention-core FLOPs only

The training step is 3 F per layer (forward + 2x backward), so "full" costs +33 %
of step FLOPs and "selective" only the fraction of F spent in QK^T and PV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mlbook.systems.memory_calc import activation_bytes_per_layer


@dataclass(frozen=True)
class CheckpointPlan:
    strategy: str
    memory_bytes: float
    extra_flops_fraction: float  # relative to a 3F training step


def attention_core_fraction(s: int, h: int, d_ff_mult: float = 4.0) -> float:
    """Fraction of a layer's forward FLOPs in QK^T + PV: 4 s h / (2 * 4 h^2 + 2 * 2 h d_ff + 4 s h)."""
    proj = 2 * 4 * h * h  # Q, K, V, O projections
    mlp = 2 * 2 * h * (d_ff_mult * h)
    attn = 4 * s * h
    return attn / (proj + mlp + attn)


def checkpoint_plans(s: int, b: int, h: int, a: int, L: int) -> list[CheckpointPlan]:
    """Memory (bytes) and extra step-FLOP fraction for every strategy."""
    A = activation_bytes_per_layer(s, b, h, a)
    I = 2.0 * s * b * h
    A_sel = activation_bytes_per_layer(s, b, h, a, recompute="selective")
    c = math.ceil(math.sqrt(L))
    frac_attn = attention_core_fraction(s, h)
    return [
        CheckpointPlan("none", L * A, 0.0),
        CheckpointPlan("selective", L * A_sel, frac_attn / 3.0),
        CheckpointPlan("sqrt", c * I + math.ceil(L / c) * A, 1.0 / 3.0),
        CheckpointPlan("full", L * I + A, 1.0 / 3.0),
    ]
