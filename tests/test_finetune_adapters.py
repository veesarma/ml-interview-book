"""Tests for bottleneck adapters (Part VI, chapter 6)."""

import torch
import torch.nn as nn

from mlbook.finetune.adapters import AdaptedSublayer, BottleneckAdapter


def test_adapter_is_identity_at_init():
    a = BottleneckAdapter(d_model=16, r=4)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(a(x), x)
    a.up.weight.data.normal_()
    assert not torch.allclose(a(x), x)


def test_adapted_sublayer_freezes_body_trains_adapter():
    body = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16))
    wrapped = AdaptedSublayer(body, d_model=16, r=4)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(wrapped(x), body(x))
    wrapped(x).sum().backward()
    assert all(p.grad is None for p in body.parameters())
    assert wrapped.adapter.down.weight.grad is not None
    n_train = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    assert n_train == 16 * 4 + 4 + 4 * 16 + 16
