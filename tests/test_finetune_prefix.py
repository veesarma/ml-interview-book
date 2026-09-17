"""Tests for prefix and prompt tuning (Part VI, chapter 6)."""

import torch

from mlbook.finetune.prefix_tuning import PrefixKV, PromptTuning, attention_with_prefix
from mlbook.llm.flash_attention import standard_attention


def test_prompt_tuning_prepends_virtual_tokens():
    pt = PromptTuning(n_virtual=3, d_model=8)
    emb = torch.randn(2, 5, 8)
    out = pt(emb)
    assert out.shape == (2, 8, 8)
    assert torch.allclose(out[:, 3:], emb) and torch.allclose(out[0, :3], pt.prompt)
    assert sum(p.numel() for p in pt.parameters()) == 3 * 8


def test_prefix_attention_with_empty_prefix_equals_causal_attention():
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    pk = PrefixKV(n_prefix=0, n_kv_heads=4, d_head=8)
    prefix_k, prefix_v = pk.expand(2)
    out = attention_with_prefix(q, k, v, prefix_k, prefix_v)
    assert torch.allclose(out, standard_attention(q, k, v, causal=True), atol=1e-6)


def test_prefix_changes_output_and_respects_causality():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 6, 8) for _ in range(3))
    pk = PrefixKV(n_prefix=4, n_kv_heads=2, d_head=8)
    prefix_k, prefix_v = pk.expand(1)
    out = attention_with_prefix(q, k, v, prefix_k, prefix_v)
    assert out.shape == (1, 2, 6, 8)
    assert not torch.allclose(out, standard_attention(q, k, v, causal=True))
    # causal: truncating the sequence must not change earlier outputs
    out3 = attention_with_prefix(q[:, :, :3], k[:, :, :3], v[:, :, :3], prefix_k, prefix_v)
    assert torch.allclose(out[:, :, :3], out3, atol=1e-6)
    assert sum(p.numel() for p in pk.parameters()) == 2 * 4 * 2 * 8
