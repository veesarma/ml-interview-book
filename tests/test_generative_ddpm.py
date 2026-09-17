import math

import torch

from mlbook.generative import ddpm as D
from mlbook.generative.toy_data import distance_to_mixture_modes, gaussian_mixture_2d_labeled


def test_schedules_are_valid():
    for betas in (D.linear_beta_schedule(100), D.cosine_alpha_bar_schedule(100)):
        s = D.NoiseSchedule(betas)
        assert torch.all(betas > 0) and torch.all(betas < 1)
        assert torch.all(s.alpha_bar[1:] <= s.alpha_bar[:-1])  # monotone decreasing
        assert s.alpha_bar[-1] < 0.05                             # x_T ≈ pure noise
    cos = D.NoiseSchedule(D.cosine_alpha_bar_schedule(1000))
    lin = D.NoiseSchedule(D.linear_beta_schedule(1000))
    assert cos.alpha_bar[500] > lin.alpha_bar[500]               # cosine keeps signal longer


def test_enforce_zero_terminal_snr():
    betas = D.enforce_zero_terminal_snr(D.linear_beta_schedule(1000))
    s = D.NoiseSchedule(betas)
    assert s.alpha_bar[-1].abs() < 1e-6
    assert torch.isclose(s.alpha_bar[0], 1 - D.linear_beta_schedule(1000)[0], atol=1e-5)


def test_q_sample_marginal_matches_closed_form():
    s = D.NoiseSchedule(D.linear_beta_schedule(50))
    x0 = torch.tensor([[2.0, -1.0]]).repeat(20000, 1)             # (N, 2)
    t = torch.full((20000,), 30, dtype=torch.long)
    x_t = D.q_sample(s, x0, t, torch.randn_like(x0))
    ab = s.alpha_bar[30]
    assert torch.allclose(x_t.mean(0), ab.sqrt() * x0[0], atol=0.03)
    assert torch.allclose(x_t.var(0), (1 - ab).repeat(2), atol=0.03)
    # the closed form equals composing the one-step kernels
    x = x0[:1].repeat(20000, 1)
    for step in range(31):
        x = (1 - s.betas[step]).sqrt() * x + s.betas[step].sqrt() * torch.randn_like(x)
    assert torch.allclose(x.mean(0), ab.sqrt() * x0[0], atol=0.03)
    assert torch.allclose(x.var(0), (1 - ab).repeat(2), atol=0.03)


def test_parameterisation_conversions_round_trip():
    s = D.NoiseSchedule(D.linear_beta_schedule(100))
    x0, eps = torch.randn(8, 3), torch.randn(8, 3)
    t = torch.randint(0, 100, (8,))
    x_t = D.q_sample(s, x0, t, eps)
    assert torch.allclose(D.eps_to_x0(s, x_t, t, eps), x0, atol=1e-5)
    v = D.x0_eps_to_v(s, x0, eps, t)
    eps_r, x0_r = D.v_to_eps_x0(s, x_t, t, v)
    assert torch.allclose(eps_r, eps, atol=1e-5) and torch.allclose(x0_r, x0, atol=1e-5)


def test_posterior_mean_variance_against_bayes_rule():
    # q(x_{t-1}|x_t,x_0) ∝ q(x_t|x_{t-1}) q(x_{t-1}|x_0): check the 1-D Gaussian product formula.
    s = D.NoiseSchedule(D.linear_beta_schedule(20))
    t = torch.tensor([7]); x0 = torch.tensor([[1.5]]); x_t = torch.tensor([[0.3]])
    mean, var = D.posterior_mean_variance(s, x_t, x0, t)
    a, b, ab_prev = s.alphas[7], s.betas[7], s.alpha_bar_prev[7]
    # precision = 1/(1−ᾱ_{t−1}) + α/β ; mean = var · ( sqrt(ᾱ_{t−1}) x0/(1−ᾱ_{t−1}) + sqrt(α) x_t / β )
    prec = 1 / (1 - ab_prev) + a / b
    var_ref = 1 / prec
    mean_ref = var_ref * (ab_prev.sqrt() * x0 / (1 - ab_prev) + a.sqrt() * x_t / b)
    assert torch.isclose(var[0, 0], var_ref, rtol=1e-5) and torch.isclose(mean[0, 0], mean_ref[0, 0], rtol=1e-5)


def test_sinusoidal_embedding_shape_and_range():
    e = D.sinusoidal_embedding(torch.arange(5), 16)
    assert e.shape == (5, 16) and e.abs().max() <= 1.0
    assert torch.allclose(e[0, :8], torch.zeros(8)) and torch.allclose(e[0, 8:], torch.ones(8))


def test_eps_mlp_shapes_with_and_without_labels():
    m = D.EpsMLP(d_x=2, n_classes=4, d_hidden=32)
    x, t = torch.randn(6, 2), torch.randint(0, 10, (6,))
    assert m(x, t, torch.randint(0, 4, (6,))).shape == (6, 2)
    assert m(x, t, None).shape == (6, 2)                          # null token path


def test_ddpm_loss_and_score_from_eps():
    s = D.NoiseSchedule(D.linear_beta_schedule(10))
    m = D.EpsMLP(d_x=2, n_classes=2, d_hidden=16)
    loss = D.ddpm_loss(m, s, torch.randn(8, 2), torch.randint(0, 2, (8,)), p_uncond=0.5)
    assert loss.ndim == 0 and torch.isfinite(loss)
    eps = torch.ones(1, 2); t = torch.tensor([3])
    assert torch.allclose(D.score_from_eps(s, eps, t), -eps / s.sqrt_one_minus_alpha_bar[3])


def test_predict_eps_cfg_formula():
    s = D.NoiseSchedule(D.linear_beta_schedule(10))
    m = D.EpsMLP(d_x=2, n_classes=3, d_hidden=16)
    x, t, y = torch.randn(4, 2), torch.randint(0, 10, (4,)), torch.randint(0, 3, (4,))
    w = 2.0
    ref = (1 + w) * m(x, t, y) - w * m(x, t, None)
    assert torch.allclose(D.predict_eps_cfg(m, x, t, y, w), ref)
    assert torch.allclose(D.predict_eps_cfg(m, x, t, y, 0.0), m(x, t, y))


def _train_small_ddpm(n_steps: int = 700):
    T = 100
    s = D.NoiseSchedule(D.linear_beta_schedule(T, 1e-4, 0.05))
    x, y = gaussian_mixture_2d_labeled(2048, n_modes=4, std=0.1)
    m = D.EpsMLP(d_x=2, n_classes=4, d_hidden=96)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(n_steps):
        idx = torch.randint(0, x.shape[0], (256,))
        loss = D.ddpm_loss(m, s, x[idx], y[idx], p_uncond=0.2)
        opt.zero_grad(); loss.backward(); opt.step()
    return m, s


def test_samplers_produce_points_near_manifold():
    m, s = _train_small_ddpm()
    anc = D.sample_ancestral(m, s, n=300, d=2)
    ddim = D.sample_ddim(m, s, n=300, d=2, n_steps=20)
    assert distance_to_mixture_modes(anc, n_modes=4).median() < 0.5
    assert distance_to_mixture_modes(ddim, n_modes=4).median() < 0.5
    # CFG toward class 0 (mode at angle 0 → centre (2, 0)) pulls samples to that mode
    y0 = torch.zeros(300, dtype=torch.long)
    guided = D.sample_ddim(m, s, n=300, d=2, n_steps=20, y=y0, guidance_scale=2.0)
    frac_mode0 = ((guided - torch.tensor([2.0, 0.0])).norm(dim=1) < 0.6).float().mean()
    assert frac_mode0 > 0.7
