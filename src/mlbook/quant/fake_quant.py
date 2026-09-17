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
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad


def fake_quantize(x: torch.Tensor, bits: int, dim: int | None = None) -> torch.Tensor:
    """Symmetric fake quantisation with STE; the scale is computed from ``x`` (no grad to scale).

    Args:
        x: any shape; ``dim`` selects per-slice scales (e.g. dim=1 on (out, in) = per row).
    Returns:
        x̂ with the same shape, differentiable via the straight-through estimator.
    """
    q_max = 2 ** (bits - 1) - 1
    if dim is None:
        scale = x.detach().abs().max().clamp(min=1e-12) / q_max
    else:
        scale = x.detach().abs().amax(dim=dim, keepdim=True).clamp(min=1e-12) / q_max
    q = torch.clamp(RoundSTE.apply(x / scale), -q_max - 1, q_max)  # clamp's grad is 0 outside range
    return q * scale


class QATLinear(nn.Module):
    """nn.Linear whose weights (and optionally activations) pass through fake quantisation.

    The master weights stay in float; the optimiser updates them with STE gradients,
    so the model learns weights that survive rounding.  Export with ``quantize_per_group``.
    """

    def __init__(self, in_features: int, out_features: int, w_bits: int = 8, a_bits: int | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.w_bits, self.a_bits = w_bits, a_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(…, in) -> (…, out) using fake-quantised weights (per output channel)."""
        if self.a_bits is not None:
            x = fake_quantize(x, self.a_bits)  # per-tensor activation fake quant
        W_hat = fake_quantize(self.linear.weight, self.w_bits, dim=1)  # (out, in)
        return x @ W_hat.T + self.linear.bias  # (…, out)
