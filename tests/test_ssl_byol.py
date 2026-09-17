"""BYOL: EMA target, stop-gradient, predictor, and the non-collapse property."""
import torch

torch.set_num_threads(1)

import torch.nn.functional as F
from torch import nn

from mlbook.ssl import byol as B


def _encoder(d_in=6, d_h=16):
    return nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_h))


def test_byol_regression_loss_endpoints():
    p = torch.randn(4, 8)
    assert B.byol_regression_loss(p, p.clone()).item() < 1e-6      # identical directions
    assert torch.isclose(B.byol_regression_loss(p, -p), torch.tensor(4.0), atol=1e-5)  # opposite: 2−2(−1)
    assert torch.isclose(B.byol_regression_loss(p, 3.0 * p), torch.tensor(0.0), atol=1e-5)  # scale-invariant


def test_ema_update_arithmetic():
    online, target = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        online.weight.fill_(1.0); target.weight.fill_(0.0)
    B.ema_update(target, online, tau=0.9)
    assert torch.allclose(target.weight, torch.full((2, 2), 0.1))   # 0.9·0 + 0.1·1
    B.ema_update(target, online, tau=0.9)
    assert torch.allclose(target.weight, torch.full((2, 2), 0.19))


def test_target_branch_has_no_gradient_and_is_not_an_optimiser_parameter():
    model = B.BYOL(_encoder(), d_h=16, d_z=8)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    loss = model(torch.randn(4, 6), torch.randn(4, 6))
    loss.backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert model.predictor[0].weight.grad is not None
    assert model.online_encoder[0].weight.grad is not None


def test_byol_step_moves_target_toward_online_and_lowers_loss():
    torch.manual_seed(0)
    model = B.BYOL(_encoder(), d_h=16, d_z=8, tau=0.9)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    x = torch.randn(32, 6)
    before = next(model.target_encoder.parameters()).clone()
    first = model(x + 0.1 * torch.randn_like(x), x + 0.1 * torch.randn_like(x)).item()
    for _ in range(200):
        last = B.byol_step(model, opt, x + 0.1 * torch.randn_like(x), x + 0.1 * torch.randn_like(x))
    after = next(model.target_encoder.parameters())
    assert not torch.allclose(before, after)      # EMA actually moved the target
    assert last < 0.5 * first


def test_byol_representations_do_not_collapse_to_a_constant():
    """The predictor + stop-grad + EMA combination should keep feature variance alive."""
    torch.manual_seed(0)
    model = B.BYOL(_encoder(), d_h=16, d_z=8, tau=0.99)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    x = torch.randn(64, 6)
    for _ in range(300):
        B.byol_step(model, opt, x + 0.1 * torch.randn_like(x), x + 0.1 * torch.randn_like(x))
    with torch.no_grad():
        z = F.normalize(model.online_projector(model.online_encoder(x)), dim=1)  # (64, 8)
    per_dim_std = z.std(dim=0)                                   # (8,)
    assert per_dim_std.mean() > 0.02     # a collapsed encoder gives identical rows (std = 0)
