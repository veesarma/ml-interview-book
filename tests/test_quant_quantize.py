"""Tests for quantisation primitives and float formats (Part VI, chapter 5)."""

import torch

from mlbook.quant.quantize import (
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    FP32,
    apply_smoothing,
    dequantize,
    dequantize_per_group,
    int_range,
    quantization_mse,
    quantize,
    quantize_per_group,
    smoothing_scales,
)


def test_int_ranges():
    assert int_range(8, True) == (-128, 127)
    assert int_range(4, False) == (0, 15)


def test_symmetric_per_tensor_roundtrip_error_bound():
    x = torch.randn(64, 64)
    r = quantize(x, bits=8, symmetric=True)
    assert r.q.dtype == torch.int32 and r.q.min() >= -128 and r.q.max() <= 127
    err = (dequantize(r) - x).abs().max()
    assert err <= r.scale.item() / 2 + 1e-6  # round-to-nearest: |x - x̂| ≤ s/2


def test_asymmetric_uses_full_range_and_represents_zero_exactly():
    x = torch.rand(32, 32) * 3.0 + 1.0  # all positive: symmetric would waste half the codes
    r = quantize(x, bits=4, symmetric=False)
    assert r.q.min() >= 0 and r.q.max() <= 15
    assert dequantize(quantize(torch.zeros(4), 4, symmetric=False)).abs().max() == 0
    assert quantization_mse(x, 4, None, symmetric=False) < quantization_mse(x, 4, None, symmetric=True)


def test_per_channel_beats_per_tensor():
    x = torch.randn(16, 128) * torch.logspace(-2, 1, 16)[:, None]  # rows on very different scales
    per_tensor = dequantize(quantize(x, 8, dim=None))
    per_channel = dequantize(quantize(x, 8, dim=1))
    assert ((per_channel - x) ** 2).mean() < 0.5 * ((per_tensor - x) ** 2).mean()
    assert quantize(x, 8, dim=1).scale.shape == (16, 1)


def test_per_group_shapes_and_error_vs_group_size():
    W = torch.randn(32, 256)
    W[:, 5] = 20.0  # one outlier column
    r = quantize_per_group(W, bits=4, group_size=32)
    assert r.q.shape == (32, 256) and r.scale.shape == (32, 8, 1)
    assert dequantize_per_group(r, 32).shape == (32, 256)
    errs = [quantization_mse(W, 4, g) for g in (None, 256, 128, 64, 32)]
    assert errs == sorted(errs, reverse=True)  # smaller groups => lower error


def test_float_formats():
    assert FP32.max_normal == (2 - 2**-23) * 2**127
    assert FP16.max_normal == 65504.0
    assert BF16.max_normal > 3e38 and BF16.epsilon == 2**-7
    assert FP8_E4M3.max_normal == 448.0 and FP8_E5M2.max_normal == 57344.0
    assert BF16.min_normal == FP32.min_normal  # same exponent range


def test_smoothing_preserves_product():
    X, W = torch.randn(10, 8) * torch.tensor([50.0, 1, 1, 1, 1, 1, 1, 1]), torch.randn(6, 8)
    s = smoothing_scales(X.abs().amax(0), W.abs().amax(0), alpha=0.5)
    X_s, W_s = apply_smoothing(X, W, s)
    assert torch.allclose(X_s @ W_s.T, X @ W.T, atol=1e-4)
    assert X_s.abs().amax(0)[0] < X.abs().amax(0)[0]  # the activation outlier got tamer
