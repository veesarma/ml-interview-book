"""Tests for the blockwise FlashAttention forward (Part VI, chapter 4)."""

import torch

from mlbook.llm.flash_attention import (
    attention_hbm_bytes_flash,
    attention_hbm_bytes_standard,
    flash_attention_forward,
    online_softmax,
    standard_attention,
)


def test_online_softmax_matches_softmax():
    x = torch.randn(37) * 5
    assert torch.allclose(online_softmax(x, chunk=8), torch.softmax(x, 0), atol=1e-6)


def test_flash_forward_equals_standard_attention():
    q, k, v = (torch.randn(2, 3, 50, 16) for _ in range(3))
    out, lse = flash_attention_forward(q, k, v, block_q=16, block_kv=8, causal=False)
    assert torch.allclose(out, standard_attention(q, k, v, causal=False), atol=1e-5)
    s = q @ k.transpose(-2, -1) / 4.0
    assert torch.allclose(lse, torch.logsumexp(s, dim=-1), atol=1e-5)


def test_flash_forward_causal_with_ragged_blocks():
    q, k, v = (torch.randn(1, 2, 45, 8) for _ in range(3))
    out, _ = flash_attention_forward(q, k, v, block_q=7, block_kv=11, causal=True)
    assert torch.allclose(out, standard_attention(q, k, v, causal=True), atol=1e-5)


def test_hbm_traffic_scales_quadratically_for_standard_only():
    std_1k, std_4k = attention_hbm_bytes_standard(1024, 128), attention_hbm_bytes_standard(4096, 128)
    assert 14 < std_4k / std_1k < 16.5  # ~16x for 4x longer sequence
    flash_1k = attention_hbm_bytes_flash(1024, 128, sram_bytes=100_000)
    assert flash_1k < std_1k
