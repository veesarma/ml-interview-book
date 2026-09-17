"""Tests for QAT fake quantisation with the straight-through estimator (Part VI, chapter 5)."""

import torch

from mlbook.quant.fake_quant import QATLinear, RoundSTE, fake_quantize
from mlbook.quant.quantize import dequantize, quantize


def test_round_ste_forward_rounds_backward_identity():
    x = torch.tensor([0.4, 1.6, -2.3], requires_grad=True)
    y = RoundSTE.apply(x)
    assert torch.equal(y, torch.tensor([0.0, 2.0, -2.0]))
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(3))


def test_fake_quantize_matches_quantize_dequantize_and_passes_gradient():
    x = torch.randn(6, 5, requires_grad=True)
    y = fake_quantize(x, bits=8, dim=1)
    ref = dequantize(quantize(x.detach(), 8, symmetric=True, dim=1))
    assert torch.allclose(y, ref, atol=1e-6)
    (y * torch.arange(5.0)).sum().backward()
    assert torch.allclose(x.grad, torch.arange(5.0).expand(6, 5))  # STE: d y / d x = 1 in range


def test_qat_linear_trains_toward_a_target():
    torch.manual_seed(0)
    layer = QATLinear(8, 4, w_bits=4)
    x = torch.randn(64, 8)
    target = torch.randn(64, 4)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    loss0 = ((layer(x) - target) ** 2).mean().item()
    for _ in range(100):
        loss = ((layer(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < loss0 * 0.7
    assert layer.linear.weight.grad is not None  # gradients reached the float master weights
