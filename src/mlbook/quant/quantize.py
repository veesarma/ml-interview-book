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

    q: torch.Tensor  # integer codes, same shape as x (stored as int32 for clarity)
    scale: torch.Tensor  # broadcastable to x
    zero: torch.Tensor  # broadcastable to x (all zeros when symmetric)
    bits: int
    symmetric: bool


def int_range(bits: int, symmetric: bool) -> tuple[int, int]:
    """(q_min, q_max): symmetric [-2^(b-1), 2^(b-1)-1], asymmetric [0, 2^b - 1]."""
    if symmetric:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _reduce(x: torch.Tensor, fn: str, dim: int | None) -> torch.Tensor:
    """min/max/absmax over ``dim`` (keepdim) or over the whole tensor."""
    if fn == "absmax":
        x = x.abs()
        fn = "max"
    if dim is None:
        return getattr(x, fn)().reshape(1)
    return getattr(x, fn)(dim=dim, keepdim=True).values


def quantize(x: torch.Tensor, bits: int, symmetric: bool = True, dim: int | None = None) -> QuantResult:
    """Affine quantisation with one scale per tensor (``dim=None``) or per slice along ``dim``.

    Args:
        x: any shape.
        dim: reduction axis for the statistics; ``dim=1`` on a (out, in) weight gives
            per-output-channel scales of shape (out, 1).
    """
    q_min, q_max = int_range(bits, symmetric)
    if symmetric:
        scale = _reduce(x, "absmax", dim) / q_max  # (…,1) or (1,)
        scale = scale.clamp(min=1e-12)
        zero = torch.zeros_like(scale)
    else:
        x_min = _reduce(x, "min", dim).clamp(max=0.0)  # include 0 so zero is representable exactly
        x_max = _reduce(x, "max", dim).clamp(min=0.0)
        scale = ((x_max - x_min) / (q_max - q_min)).clamp(min=1e-12)
        zero = torch.round(-x_min / scale)  # integer offset so that x_min -> q_min
    q = torch.clamp(torch.round(x / scale) + zero, q_min, q_max).to(torch.int32)
    return QuantResult(q, scale, zero, bits, symmetric)


def dequantize(r: QuantResult) -> torch.Tensor:
    """x̂ = scale · (q − zero), same shape as the original."""
    return r.scale * (r.q.to(r.scale.dtype) - r.zero)


def quantize_per_group(W: torch.Tensor, bits: int, group_size: int, symmetric: bool = True) -> QuantResult:
    """Per-group quantisation of a (out, in) weight: one (scale, zero) per ``group_size`` inputs.

    Returns codes of shape (out, in) and scale/zero of shape (out, in // group_size, 1)
    broadcast over the grouped view (out, in // group_size, group_size).
    """
    out_f, in_f = W.shape
    if in_f % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")
    grouped = W.reshape(out_f, in_f // group_size, group_size)  # (out, n_groups, group)
    r = quantize(grouped, bits, symmetric, dim=2)  # scale: (out, n_groups, 1)
    return QuantResult(r.q.reshape(out_f, in_f), r.scale, r.zero, bits, symmetric)


def dequantize_per_group(r: QuantResult, group_size: int) -> torch.Tensor:
    """Inverse of ``quantize_per_group``: (out, in)."""
    out_f, in_f = r.q.shape
    grouped = r.q.reshape(out_f, in_f // group_size, group_size).to(r.scale.dtype)  # (out, n_groups, group)
    return (r.scale * (grouped - r.zero)).reshape(out_f, in_f)


def quantization_mse(W: torch.Tensor, bits: int, group_size: int | None, symmetric: bool = True) -> float:
    """Mean squared reconstruction error of round-to-nearest at a given granularity."""
    if group_size is None:
        W_hat = dequantize(quantize(W, bits, symmetric, dim=None))
    else:
        W_hat = dequantize_per_group(quantize_per_group(W, bits, group_size, symmetric), group_size)
    return float(((W - W_hat) ** 2).mean())


# ---------------------------------------------------------------------------
# Floating-point formats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloatFormat:
    """IEEE-style format with ``exp_bits`` exponent bits and ``mant_bits`` mantissa bits."""

    name: str
    exp_bits: int
    mant_bits: int
    fp8_e4m3_special: bool = False  # E4M3 (OCP/NVIDIA) drops inf and keeps one NaN to reach 448

    @property
    def bias(self) -> int:
        return 2 ** (self.exp_bits - 1) - 1

    @property
    def max_normal(self) -> float:
        """Largest finite value: (2 − 2^−m) · 2^(e_max − bias)."""
        if self.fp8_e4m3_special:
            return 448.0  # 1.75 · 2^8: the all-ones exponent is reused for finite values
        e_max = 2**self.exp_bits - 2  # all-ones exponent is reserved for inf/NaN
        return (2.0 - 2.0**-self.mant_bits) * 2.0 ** (e_max - self.bias)

    @property
    def min_normal(self) -> float:
        return 2.0 ** (1 - self.bias)

    @property
    def epsilon(self) -> float:
        """Spacing between 1 and the next representable number: 2^−m (relative precision)."""
        return 2.0**-self.mant_bits


FP32 = FloatFormat("FP32", exp_bits=8, mant_bits=23)
FP16 = FloatFormat("FP16", exp_bits=5, mant_bits=10)
BF16 = FloatFormat("BF16", exp_bits=8, mant_bits=7)
FP8_E4M3 = FloatFormat("FP8 E4M3", exp_bits=4, mant_bits=3, fp8_e4m3_special=True)
FP8_E5M2 = FloatFormat("FP8 E5M2", exp_bits=5, mant_bits=2)


# ---------------------------------------------------------------------------
# SmoothQuant / AWQ style per-input-channel smoothing
# ---------------------------------------------------------------------------


def smoothing_scales(act_absmax: torch.Tensor, weight_absmax: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant scale per input channel: s_j = max|X_j|^α / max|W_j|^(1−α).

    Args:
        act_absmax:    (in,) per-input-channel activation magnitude from calibration data.
        weight_absmax: (in,) per-input-channel weight magnitude (max over output rows).
    Returns:
        (in,) positive scales.  AWQ uses the same shape of transform with s_j = act_mean_j^α
        (searching α) and chooses α by reconstruction error.
    """
    return (act_absmax.clamp(min=1e-5) ** alpha) / (weight_absmax.clamp(min=1e-5) ** (1 - alpha))


def apply_smoothing(X: torch.Tensor, W: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X / s, W · s) so that (X/s)(W·s)^T = X W^T exactly.

    Args:
        X: (N, in) activations, W: (out, in) weight, s: (in,).
    """
    return X / s[None, :], W * s[None, :]
