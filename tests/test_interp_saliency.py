import torch
from torch import nn

from mlbook.interp import saliency as sal


def test_vanilla_gradient_of_linear_model_is_its_weight_row():
    torch.manual_seed(0)
    lin = nn.Linear(6, 3)
    x = torch.randn(6)
    g = sal.vanilla_gradient(lin, x, target=2)
    assert torch.allclose(g, lin.weight[2].detach())


def test_smoothgrad_of_linear_model_equals_vanilla_gradient():
    torch.manual_seed(0)
    lin = nn.Linear(6, 3)
    x = torch.randn(6)
    assert torch.allclose(sal.smoothgrad(lin, x, 1, n_samples=8), sal.vanilla_gradient(lin, x, 1), atol=1e-6)


def test_saliency_map_collapses_channels_and_normalises():
    g = torch.tensor([[[1.0, -3.0]], [[2.0, 0.5]]])  # (2, 1, 2)
    m = sal.saliency_map(g)
    assert m.shape == (1, 2) and torch.allclose(m, torch.tensor([[2 / 3, 1.0]]))
