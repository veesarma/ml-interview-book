"""Tests for grouped-query attention (Part VI, chapter 3)."""

import torch
import torch.nn.functional as F

from mlbook.llm.gqa import GroupedQueryAttention, kv_cache_reduction_factor


def _reference(gqa: GroupedQueryAttention, x: torch.Tensor) -> torch.Tensor:
    """MHA reference via F.scaled_dot_product_attention on explicitly expanded K/V."""
    B, T, _ = x.shape
    q, k, v = gqa.project(x)
    k = torch.cat([k[:, i : i + 1].expand(-1, gqa.group, -1, -1) for i in range(gqa.n_kv_heads)], dim=1)
    v = torch.cat([v[:, i : i + 1].expand(-1, gqa.group, -1, -1) for i in range(gqa.n_kv_heads)], dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, d_head)
    return gqa.o_proj(out.transpose(1, 2).reshape(B, T, -1))


def test_gqa_equals_mha_when_kv_heads_equal_heads():
    gqa = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=4)
    x = torch.randn(2, 7, 32)
    assert torch.allclose(gqa(x), _reference(gqa, x), atol=1e-5)
    assert gqa.group == 1 and gqa.kv_bytes_per_token() == 2 * 4 * 8 * 2


def test_gqa_equals_mqa_when_one_kv_head():
    gqa = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=1)
    x = torch.randn(2, 7, 32)
    assert torch.allclose(gqa(x), _reference(gqa, x), atol=1e-5)
    # MQA: all four heads must read the same K -> the same as attending with a single shared head
    q, k, v = gqa.project(x)
    assert k.shape == (2, 1, 7, 8)


def test_gqa_grouping_maps_head_h_to_kv_head_h_div_group():
    gqa = GroupedQueryAttention(d_model=48, n_heads=6, n_kv_heads=2)
    x = torch.randn(1, 5, 48)
    assert torch.allclose(gqa(x), _reference(gqa, x), atol=1e-5)
    assert gqa.group == 3
    assert kv_cache_reduction_factor(6, 2) == 3.0
    assert kv_cache_reduction_factor(64, 8) == 8.0


def test_gqa_is_causal():
    gqa = GroupedQueryAttention(d_model=16, n_heads=2, n_kv_heads=1)
    x = torch.randn(1, 6, 16)
    y_full = gqa(x)
    y_prefix = gqa(x[:, :3])
    assert torch.allclose(y_full[:, :3], y_prefix, atol=1e-6)  # future tokens do not change the past
