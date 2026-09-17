import torch
from torch import nn

from mlbook.systems import grad_accumulation as ga


def _data(n=32, d=8):
    g = torch.Generator().manual_seed(1)
    return torch.randn(n, d, generator=g), torch.randn(n, 1, generator=g)


def test_accumulated_grads_equal_full_batch_with_layernorm():
    X, y = _data()
    model = ga.make_mlp("layernorm")
    full = ga.full_batch_grads(model, nn.functional.mse_loss, X, y)
    acc = ga.accumulated_grads(model, nn.functional.mse_loss, X, y, k=4)
    assert ga.max_abs_diff(full, acc) < 1e-6


def test_accumulated_grads_differ_with_batchnorm():
    X, y = _data()
    model = ga.make_mlp("batchnorm")
    full = ga.full_batch_grads(model, nn.functional.mse_loss, X, y)
    acc = ga.accumulated_grads(model, nn.functional.mse_loss, X, y, k=4)
    assert ga.max_abs_diff(full, acc) > 1e-4
