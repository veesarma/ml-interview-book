"""Sinusoidal, learned, T5 relative bias, RoPE (relative-position property), ALiBi."""
import math

import torch

from mlbook.transformer.positional import (
    LearnedPositionalEmbedding,
    RelativePositionBias,
    RotaryEmbedding,
    alibi_bias,
    alibi_slopes,
    apply_rotary,
    relative_position_bucket,
    rotate_half,
    sinusoidal_positional_encoding,
)


def test_sinusoidal_values_and_linear_offset_property():
    pe = sinusoidal_positional_encoding(50, 8)
    assert pe.shape == (50, 8)
    assert math.isclose(pe[3, 0].item(), math.sin(3.0), abs_tol=1e-6)
    assert math.isclose(pe[3, 1].item(), math.cos(3.0), abs_tol=1e-6)
    # PE[t + k] is a rotation of PE[t] by a fixed matrix depending on k only: check pair 0 with k = 5
    k = 5
    w = 1.0
    R = torch.tensor([[math.cos(k * w), -math.sin(k * w)], [math.sin(k * w), math.cos(k * w)]])  # [sin t, cos t] @ R = [sin(t+k), cos(t+k)]
    for t in (0, 7, 20):
        torch.testing.assert_close(pe[t + k, 0:2], pe[t, 0:2] @ R, atol=1e-5, rtol=1e-5)


def test_learned_positional_embedding_offset():
    emb = LearnedPositionalEmbedding(10, 4)
    full = emb(10)
    part = emb(3, offset=5)
    torch.testing.assert_close(part, full[:, 5:8])


def test_relative_position_bucket_properties():
    rel = torch.arange(-40, 41)[None, :]  # key - query
    b = relative_position_bucket(rel, bidirectional=True, num_buckets=32, max_distance=128)
    assert b.min() >= 0 and b.max() < 32
    assert b[0, 40] == 0  # distance 0
    assert torch.equal(b[0, 40 - 7 : 40], torch.arange(7, 0, -1))  # exact small buckets for past keys
    causal = relative_position_bucket(rel, bidirectional=False, num_buckets=32, max_distance=128)
    assert torch.all(causal[0, 41:] == 0)  # future collapsed


def test_relative_position_bias_shape_and_translation_invariance():
    rb = RelativePositionBias(n_heads=3, num_buckets=16, max_distance=64)
    bias = rb(10, 10)
    assert bias.shape == (1, 3, 10, 10)
    torch.testing.assert_close(bias[0, :, 2, 5], bias[0, :, 6, 9])  # same offset -> same bias


def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_relative_position_property():
    """<R_m q, R_n k> depends only on m - n."""
    rope = RotaryEmbedding(d_head=16)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(m: int, n: int) -> float:
        cos_m, sin_m = rope(1, offset=m)
        cos_n, sin_n = rope(1, offset=n)
        return (apply_rotary(q, cos_m, sin_m) * apply_rotary(k, cos_n, sin_n)).sum().item()

    assert math.isclose(score(3, 1), score(10, 8), rel_tol=1e-5)
    assert math.isclose(score(3, 1), score(103, 101), rel_tol=1e-4)
    assert not math.isclose(score(3, 1), score(3, 2), rel_tol=1e-3)


def test_rope_matches_complex_multiplication():
    rope = RotaryEmbedding(d_head=8)
    x = torch.randn(1, 1, 5, 8)
    cos, sin = rope(5)
    y = apply_rotary(x, cos, sin)
    # complex view: pair (x_i, x_{i+4}) is x_i + i x_{i+4}; multiply by exp(i m theta_i)
    z = torch.complex(x[..., :4], x[..., 4:])  # (1, 1, 5, 4)
    angles = torch.arange(5)[:, None].float() * rope.inv_freq[None, :]  # (5, 4)
    z_rot = z * torch.polar(torch.ones_like(angles), angles)
    torch.testing.assert_close(y[..., :4], z_rot.real)
    torch.testing.assert_close(y[..., 4:], z_rot.imag)


def test_rope_scaling_modes():
    plain = RotaryEmbedding(8)
    linear = RotaryEmbedding(8, scaling="linear", factor=2.0)
    ntk = RotaryEmbedding(8, scaling="ntk", factor=2.0)
    c_plain, _ = plain(1, offset=4)
    c_lin, _ = linear(1, offset=8)  # position 8 / 2 == position 4 unscaled
    torch.testing.assert_close(c_plain, c_lin)
    assert ntk.inv_freq[0] == plain.inv_freq[0]  # highest frequency untouched (theta_0 = 1)
    assert ntk.inv_freq[-1] < plain.inv_freq[-1]  # lowest frequency stretched


def test_alibi_slopes_and_bias():
    torch.testing.assert_close(alibi_slopes(8), torch.tensor([2.0 ** -i for i in range(1, 9)]))
    bias = alibi_bias(2, T_q=4, T_k=4)
    assert bias.shape == (1, 2, 4, 4)
    assert bias[0, 0, 3, 3] == 0 and bias[0, 0, 3, 0] == -alibi_slopes(2)[0] * 3
    assert torch.all(bias[0, 0].triu(1) == 0)  # future is left to the mask
