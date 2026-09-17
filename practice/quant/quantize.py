# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/quant/quantize.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k quantize -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py quant/quantize --force

"""Uniform (affine) quantisation primitives (PyTorch tensors, CPU-friendly).

    symmetric:   s = max|x| / q_max,               q = clip(round(x / s), -q_max-1, q_max)
    asymmetric:  s = (x_max − x_min) / (2^b − 1),  z = round(−x_min / s),
                 q = clip(round(x / s) + z, 0, 2^b − 1)
    dequantise:  x̂ = s · (q − z)   (z = 0 for symmetric)

Granularity: one (s, z) per tensor, per output channel (row), or per group of
``group_size`` consecutive input weights (GPTQ/AWQ-style, e.g. 128).
Also here: floating-point format arithmetic (FP16/BF16/FP8) and SmoothQuant/AWQ
style per-channel smoothing, which moves quantisation difficulty from
activations into weights without changing the product.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class QuantResult:
    """Integer codes plus the affine parameters needed to dequantise."""
    q: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    symmetric: bool

def int_range(bits: int, symmetric: bool) -> tuple[int, int]:
    """(q_min, q_max): symmetric [-2^(b-1), 2^(b-1)-1], asymmetric [0, 2^b - 1]."""
    raise NotImplementedError('TODO: implement int_range (see the reference in src/mlbook)')

def _reduce(x: torch.Tensor, fn: str, dim: int | None) -> torch.Tensor:
    """min/max/absmax over ``dim`` (keepdim) or over the whole tensor."""
    raise NotImplementedError('TODO: implement _reduce (see the reference in src/mlbook)')

def quantize(x: torch.Tensor, bits: int, symmetric: bool=True, dim: int | None=None) -> QuantResult:
    """Affine quantisation with one scale per tensor (``dim=None``) or per slice along ``dim``.

    Args:
        x: any shape.
        dim: reduction axis for the statistics; ``dim=1`` on a (out, in) weight gives
            per-output-channel scales of shape (out, 1).
    """
    raise NotImplementedError('TODO: implement quantize (see the reference in src/mlbook)')

def dequantize(r: QuantResult) -> torch.Tensor:
    """x̂ = scale · (q − zero), same shape as the original."""
    raise NotImplementedError('TODO: implement dequantize (see the reference in src/mlbook)')

def quantize_per_group(W: torch.Tensor, bits: int, group_size: int, symmetric: bool=True) -> QuantResult:
    """Per-group quantisation of a (out, in) weight: one (scale, zero) per ``group_size`` inputs.

    Returns codes of shape (out, in) and scale/zero of shape (out, in // group_size, 1)
    broadcast over the grouped view (out, in // group_size, group_size).
    """
    raise NotImplementedError('TODO: implement quantize_per_group (see the reference in src/mlbook)')

def dequantize_per_group(r: QuantResult, group_size: int) -> torch.Tensor:
    """Inverse of ``quantize_per_group``: (out, in)."""
    raise NotImplementedError('TODO: implement dequantize_per_group (see the reference in src/mlbook)')

def quantization_mse(W: torch.Tensor, bits: int, group_size: int | None, symmetric: bool=True) -> float:
    """Mean squared reconstruction error of round-to-nearest at a given granularity."""
    raise NotImplementedError('TODO: implement quantization_mse (see the reference in src/mlbook)')

@dataclass(frozen=True)
class FloatFormat:
    """IEEE-style format with ``exp_bits`` exponent bits and ``mant_bits`` mantissa bits."""
    name: str
    exp_bits: int
    mant_bits: int
    fp8_e4m3_special: bool = False

    @property
    def bias(self) -> int:
        raise NotImplementedError('TODO: implement bias (see the reference in src/mlbook)')

    @property
    def max_normal(self) -> float:
        """Largest finite value: (2 − 2^−m) · 2^(e_max − bias)."""
        raise NotImplementedError('TODO: implement max_normal (see the reference in src/mlbook)')

    @property
    def min_normal(self) -> float:
        raise NotImplementedError('TODO: implement min_normal (see the reference in src/mlbook)')

    @property
    def epsilon(self) -> float:
        """Spacing between 1 and the next representable number: 2^−m (relative precision)."""
        raise NotImplementedError('TODO: implement epsilon (see the reference in src/mlbook)')
FP32 = FloatFormat('FP32', exp_bits=8, mant_bits=23)
FP16 = FloatFormat('FP16', exp_bits=5, mant_bits=10)
BF16 = FloatFormat('BF16', exp_bits=8, mant_bits=7)
FP8_E4M3 = FloatFormat('FP8 E4M3', exp_bits=4, mant_bits=3, fp8_e4m3_special=True)
FP8_E5M2 = FloatFormat('FP8 E5M2', exp_bits=5, mant_bits=2)

def smoothing_scales(act_absmax: torch.Tensor, weight_absmax: torch.Tensor, alpha: float=0.5) -> torch.Tensor:
    """SmoothQuant scale per input channel: s_j = max|X_j|^α / max|W_j|^(1−α).

    Args:
        act_absmax:    (in,) per-input-channel activation magnitude from calibration data.
        weight_absmax: (in,) per-input-channel weight magnitude (max over output rows).
    Returns:
        (in,) positive scales.  AWQ uses the same shape of transform with s_j = act_mean_j^α
        (searching α) and chooses α by reconstruction error.
    """
    raise NotImplementedError('TODO: implement smoothing_scales (see the reference in src/mlbook)')

def apply_smoothing(X: torch.Tensor, W: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X / s, W · s) so that (X/s)(W·s)^T = X W^T exactly.

    Args:
        X: (N, in) activations, W: (out, in) weight, s: (in,).
    """
    raise NotImplementedError('TODO: implement apply_smoothing (see the reference in src/mlbook)')
