import torch
from torch import nn

from mlbook.interp import integrated_gradients as ig


def test_integrated_gradients_linear_model_is_w_times_delta():
    torch.manual_seed(0)
    lin = nn.Linear(5, 2)
    x, base = torch.randn(5), torch.zeros(5)
    a = ig.integrated_gradients(lin, x, base, target=0, steps=16)
    assert torch.allclose(a, lin.weight[0].detach() * x, atol=1e-6)


def test_completeness_gap_shrinks_with_steps_on_nonlinear_model():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 3))
    x, base = torch.randn(5) * 2, torch.zeros(5)
    g8, g256 = ig.completeness_gap(net, x, base, 1, steps=8), ig.completeness_gap(net, x, base, 1, steps=256)
    assert g256 < g8 and g256 < 1e-3
