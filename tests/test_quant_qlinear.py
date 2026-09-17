"""Tests for quantised Linear, INT4 packing, LLM.int8 decomposition and GPTQ (Part VI, chapter 5)."""

import torch
import torch.nn as nn

from mlbook.quant.qlinear import QuantizedLinear, gptq_quantize, int8_matmul_with_outliers, pack_int4, unpack_int4


def test_int4_pack_unpack_roundtrip():
    q = torch.randint(-8, 8, (5, 12), dtype=torch.int32)
    packed = pack_int4(q)
    assert packed.shape == (5, 6) and packed.dtype == torch.uint8
    assert torch.equal(unpack_int4(packed), q)


def test_quantized_linear_matches_float_linear():
    lin = nn.Linear(256, 64)
    x = torch.randn(8, 256)
    y = lin(x)
    q8 = QuantizedLinear.from_linear(lin, bits=8, group_size=64)
    q4 = QuantizedLinear.from_linear(lin, bits=4, group_size=32)
    err8 = (q8(x) - y).abs().mean() / y.abs().mean()
    err4 = (q4(x) - y).abs().mean() / y.abs().mean()
    assert err8 < 0.01 and err4 < 0.1 and err8 < err4
    assert q4.codes.shape == (64, 128)  # two nibbles per byte
    assert q4.weight_bytes() < lin.weight.numel() * 2  # smaller than fp16


def test_error_decreases_with_smaller_groups():
    lin = nn.Linear(512, 32)
    with torch.no_grad():
        lin.weight[:, 3] *= 30.0  # outlier input channel
    x = torch.randn(16, 512)
    errs = []
    for g in (512, 128, 32):
        ql = QuantizedLinear.from_linear(lin, bits=4, group_size=g)
        errs.append((ql(x) - lin(x)).pow(2).mean().item())
    assert errs[0] > errs[1] > errs[2]


def test_int8_with_outliers_is_close_and_better_than_plain_int8():
    X = torch.randn(32, 64)
    X[:, 7] *= 60.0  # an outlier feature dimension, as in >6.7B models
    W = torch.randn(16, 64)
    ref = X @ W.T
    mixed = int8_matmul_with_outliers(X, W, threshold=6.0)
    plain = int8_matmul_with_outliers(X, W, threshold=1e9)  # nothing treated as outlier
    assert (mixed - ref).abs().mean() < (plain - ref).abs().mean()
    assert (mixed - ref).abs().mean() / ref.abs().mean() < 0.02


def test_gptq_beats_round_to_nearest_on_correlated_inputs():
    torch.manual_seed(0)
    n, in_f, out_f = 512, 64, 32
    basis = torch.randn(in_f, 8)
    X = torch.randn(n, 8) @ basis.T + 0.1 * torch.randn(n, in_f)  # low-rank, correlated features
    W = torch.randn(out_f, in_f)
    H = X.T @ X / n
    W_gptq = gptq_quantize(W, H, bits=3, group_size=64)
    from mlbook.quant.quantize import dequantize_per_group, quantize_per_group

    W_rtn = dequantize_per_group(quantize_per_group(W, bits=3, group_size=64), 64)
    err_gptq = ((X @ W_gptq.T - X @ W.T) ** 2).mean()
    err_rtn = ((X @ W_rtn.T - X @ W.T) ** 2).mean()
    assert err_gptq < 0.7 * err_rtn
