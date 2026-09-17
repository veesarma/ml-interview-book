import torch

from mlbook.generative import flow_matching as FM
from mlbook.generative.toy_data import distance_to_mixture_modes, gaussian_mixture_2d


def test_linear_path_endpoints_and_velocity():
    x0, x1 = torch.randn(5, 2), torch.randn(5, 2)
    xt0, u = FM.linear_path(x0, x1, torch.zeros(5))
    xt1, _ = FM.linear_path(x0, x1, torch.ones(5))
    assert torch.allclose(xt0, x0) and torch.allclose(xt1, x1) and torch.allclose(u, x1 - x0)
    # velocity is the time derivative of the path
    xt_a, _ = FM.linear_path(x0, x1, torch.full((5,), 0.3))
    xt_b, _ = FM.linear_path(x0, x1, torch.full((5,), 0.3 + 1e-3))
    assert torch.allclose((xt_b - xt_a) / 1e-3, u, atol=1e-3)


def test_gaussian_path_reduces_to_linear_when_sigma_min_zero():
    x0, x1, t = torch.randn(5, 2), torch.randn(5, 2), torch.rand(5)
    xa, ua = FM.gaussian_path(x1, x0, t, sigma_min=0.0)
    xb, ub = FM.linear_path(x0, x1, t)
    assert torch.allclose(xa, xb) and torch.allclose(ua, ub)


def test_cfm_loss_is_finite_scalar():
    m = FM.VelocityMLP(d_x=2, d_hidden=16)
    loss = FM.cfm_loss(m, torch.randn(8, 2))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_divergence_exact_matches_hutchinson_and_autograd_jacobian():
    m = FM.VelocityMLP(d_x=2, d_hidden=16)
    x, t = torch.randn(3, 2), torch.rand(3)
    exact = FM.divergence_exact(m, x, t)
    jac = torch.autograd.functional.jacobian(lambda xx: m(xx, t[:1]), x[:1])  # (1, 2, 1, 2)
    assert torch.isclose(exact[0], jac[0, 0, 0, 0] + jac[0, 1, 0, 1], atol=1e-5)
    hutch = FM.divergence_hutchinson(m, x, t, n_probes=2000)
    assert torch.allclose(hutch, exact, atol=0.1)


def test_sample_euler_trajectory_shape():
    m = FM.VelocityMLP(d_x=2, d_hidden=16)
    traj = FM.sample_euler(m, n=4, d=2, n_steps=5, return_trajectory=True)
    assert traj.shape == (6, 4, 2)
    assert FM.sample_midpoint(m, n=4, d=2, n_steps=3).shape == (4, 2)


def test_flow_matching_learns_toy_distribution_with_few_euler_steps():
    x = gaussian_mixture_2d(2048, n_modes=4, std=0.1)
    m = FM.VelocityMLP(d_x=2, d_hidden=96)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(700):
        idx = torch.randint(0, x.shape[0], (256,))
        loss = FM.cfm_loss(m, x[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    samples = FM.sample_euler(m, n=300, d=2, n_steps=8)
    assert distance_to_mixture_modes(samples, n_modes=4).median() < 0.5
