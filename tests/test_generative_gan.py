import math

import torch

from mlbook.generative import gan as G
from mlbook.generative.toy_data import distance_to_mixture_modes, gaussian_mixture_2d


def test_d_loss_standard_at_chance_is_log4():
    zeros = torch.zeros(10)  # logit 0 → D = ½ everywhere
    assert torch.isclose(G.d_loss_standard(zeros, zeros), torch.tensor(math.log(4.0)))


def test_g_loss_non_saturating_has_gradient_when_d_is_confident():
    logit = torch.tensor([-10.0], requires_grad=True)  # D(G(z)) ≈ 0: D is confident the sample is fake
    G.g_loss_saturating(logit).backward()
    sat_grad = logit.grad.clone(); logit.grad = None
    G.g_loss_non_saturating(logit).backward()
    assert sat_grad.abs().item() < 1e-3          # saturates
    assert logit.grad.abs().item() > 0.9         # ≈ −1: still learns


def test_wasserstein_losses_and_gradient_penalty():
    real, fake = torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0])
    assert torch.isclose(G.d_loss_wasserstein(real, fake), torch.tensor(-2.0))
    assert torch.isclose(G.g_loss_wasserstein(fake), torch.tensor(-0.5))
    # a linear critic f(x) = 3·x_1 has gradient norm 3 everywhere → penalty (3 − 1)² = 4
    critic = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3.0, 0.0]]))
    gp = G.gradient_penalty(lambda x: critic(x)[:, 0], torch.randn(8, 2), torch.randn(8, 2))
    assert torch.isclose(gp, torch.tensor(4.0), atol=1e-5)


def test_spectral_norm_power_iteration_matches_svd():
    w = torch.randn(6, 4)
    u = torch.randn(6)
    sigma, u = G.spectral_norm_power_iteration(w, u, n_iter=100)
    assert torch.isclose(sigma, torch.linalg.matrix_norm(w, ord=2), rtol=1e-4)


def test_gan_train_step_losses_behave_sanely():
    x = gaussian_mixture_2d(2048, n_modes=4, std=0.1)
    gen, disc = G.Generator(d_z=2, d_x=2, d_hidden=64), G.Discriminator(d_x=2, d_hidden=64)
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-3, betas=(0.5, 0.999))
    d_losses, g_losses = [], []
    for step in range(600):
        idx = torch.randint(0, x.shape[0], (128,))
        d_l, g_l = G.gan_train_step(gen, disc, opt_g, opt_d, x[idx], d_z=2)
        d_losses.append(d_l); g_losses.append(g_l)
    assert all(math.isfinite(v) for v in d_losses + g_losses)
    # D never gets a free win: its loss stays well above 0 (it would be ~0 if G collapsed to junk)
    assert sum(d_losses[-100:]) / 100 > 0.3
    # samples land near the data manifold
    with torch.no_grad():
        samples = gen(torch.randn(500, 2))
    assert distance_to_mixture_modes(samples, n_modes=4).median() < 0.6
