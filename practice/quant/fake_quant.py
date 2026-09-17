# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/quant/fake_quant.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k fake_quant -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py quant/fake_quant --force

"""Quantisation-aware training: fake quantisation with the straight-through estimator.

Forward:  x̂ = s · clip(round(x / s), q_min, q_max)      (quantise, then dequantise)
Backward: ∂L/∂x := ∂L/∂x̂ · 1[q_min ≤ x/s ≤ q_max]        (round has zero gradient
          almost everywhere, so we pretend it is the identity inside the clip range)
"""
from __future__ import annotations
import torch
import torch.nn as nn

class RoundSTE(torch.autograd.Function):
    """round(x) in the forward pass, identity gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

def fake_quantize(x: torch.Tensor, bits: int, dim: int | None=None) -> torch.Tensor:
    """Symmetric fake quantisation with STE; the scale is computed from ``x`` (no grad to scale).

    Args:
        x: any shape; ``dim`` selects per-slice scales (e.g. dim=1 on (out, in) = per row).
    Returns:
        x̂ with the same shape, differentiable via the straight-through estimator.
    """
    raise NotImplementedError('TODO: implement fake_quantize (see the reference in src/mlbook)')

class QATLinear(nn.Module):
    """nn.Linear whose weights (and optionally activations) pass through fake quantisation.

    The master weights stay in float; the optimiser updates them with STE gradients,
    so the model learns weights that survive rounding.  Export with ``quantize_per_group``.
    """

    def __init__(self, in_features: int, out_features: int, w_bits: int=8, a_bits: int | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(…, in) -> (…, out) using fake-quantised weights (per output channel)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
