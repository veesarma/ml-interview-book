"""Tests for LoRA (Part VI, chapter 6)."""

import torch
import torch.nn as nn

from mlbook.finetune.lora import (
    LoRALinear,
    MultiLoRALinear,
    apply_lora,
    count_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
)
from mlbook.llm.gqa import GroupedQueryAttention


def test_lora_is_identity_at_init_then_changes_after_update():
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8.0)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x))  # B = 0
    lora.lora_B.data.normal_()
    assert not torch.allclose(lora(x), base(x))
    assert lora.delta_weight().shape == (8, 16)


def test_merge_equals_unmerged_and_unmerge_restores():
    lora = LoRALinear(nn.Linear(16, 8), r=4, alpha=16.0)
    lora.lora_B.data.normal_()
    x = torch.randn(5, 16)
    y_unmerged = lora(x)
    W0 = lora.base.weight.clone()
    lora.merge()
    assert lora.merged and torch.allclose(lora(x), y_unmerged, atol=1e-6)
    assert not torch.allclose(lora.base.weight, W0)
    lora.unmerge()
    assert torch.allclose(lora.base.weight, W0, atol=1e-6) and torch.allclose(lora(x), y_unmerged, atol=1e-6)


def test_apply_lora_to_attention_and_only_lora_trainable():
    attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2)
    x = torch.randn(2, 6, 32)
    y_before = attn(x)
    wrapped = apply_lora(attn, r=2, alpha=4.0, target_names=("q_proj", "v_proj"))
    assert wrapped == ["q_proj", "v_proj"]
    assert isinstance(attn.q_proj, LoRALinear) and isinstance(attn.k_proj, nn.Linear)
    mark_only_lora_trainable(attn)
    trainable = [n for n, p in attn.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable) and len(trainable) == 4
    n_train, n_total = count_parameters(attn)
    assert n_train == (2 * 32 + 32 * 2) + (2 * 32 + 16 * 2) and n_train < 0.1 * n_total  # q: out=32, v: out=16
    assert torch.allclose(attn(x), y_before)  # still the base model at init
    attn(x).sum().backward()
    assert attn.q_proj.lora_B.grad is not None and attn.q_proj.base.weight.grad is None
    assert set(lora_state_dict(attn)) == {"q_proj.lora_A", "q_proj.lora_B", "v_proj.lora_A", "v_proj.lora_B"}


def test_multi_lora_selects_adapter_per_example():
    base = nn.Linear(8, 4)
    ml = MultiLoRALinear(base, n_adapters=3, r=2, alpha=2.0)
    ml.lora_B.data.normal_()
    x = torch.randn(2, 5, 8)
    y = ml(x, torch.tensor([0, 2]))
    y0 = ml(x, torch.tensor([0, 0]))
    y2 = ml(x, torch.tensor([2, 2]))
    assert torch.allclose(y[0], y0[0]) and torch.allclose(y[1], y2[1]) and not torch.allclose(y[1], y0[1])
