"""Autoencoder, VAE and VQ-VAE tests.  Each retype-by-hand symbol has its own focused test."""
import math

import torch

torch.set_num_threads(1)  # tiny tensors: multi-threading costs more than it saves

from mlbook.generative import autoencoder as ae
from mlbook.generative import vae as V
from mlbook.generative import vqvae as VQ
from mlbook.generative.toy_data import gaussian_mixture_2d


# ---------------------------------------------------------------- autoencoder.py
def test_autoencoder_shapes_and_reconstruction_improves():
    model = ae.Autoencoder(d_x=4, d_z=2, d_hidden=32)
    x = torch.randn(64, 4) @ torch.randn(4, 4)  # low-rank-ish data
    assert model.encode(x).shape == (64, 2) and model(x).shape == (64, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = ae.reconstruction_loss(x, model(x)).item()
    for _ in range(300):
        loss = ae.reconstruction_loss(x, model(x))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * before


def test_reconstruction_loss_matches_manual():
    x, xh = torch.randn(5, 3), torch.randn(5, 3)
    manual = sum(((x[i] - xh[i]) ** 2).sum() for i in range(5)) / 5
    assert torch.isclose(ae.reconstruction_loss(x, xh), manual)


def test_corruptions():
    x = torch.ones(1000, 8)
    assert abs(ae.corrupt_gaussian(x, 0.5).std().item() - 0.5) < 0.05
    dropped = (ae.corrupt_mask(x, 0.3) == 0).float().mean().item()
    assert abs(dropped - 0.3) < 0.03


def test_denoising_loss_is_finite_and_differentiable():
    model = ae.Autoencoder(4, 2)
    loss = ae.denoising_loss(model, torch.randn(16, 4), sigma=0.1)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder[0].weight.grad is not None


# ---------------------------------------------------------------- vae.py
def test_reparameterize_mean_and_std_and_gradient():
    mu = torch.tensor([[1.0, -2.0]]).requires_grad_(True)
    logvar = torch.tensor([[math.log(4.0), 0.0]]).requires_grad_(True)  # sigma = 2, 1
    z = torch.stack([V.reparameterize(mu, logvar)[0] for _ in range(4000)])  # (4000, 2)
    assert torch.allclose(z.mean(0), mu[0], atol=0.15)
    assert torch.allclose(z.std(0), torch.tensor([2.0, 1.0]), atol=0.15)
    V.reparameterize(mu, logvar).sum().backward()
    assert torch.allclose(mu.grad, torch.ones(1, 2))  # dz/dmu = 1


def test_gaussian_kl_closed_form_known_values():
    mu = torch.zeros(1, 3); logvar = torch.zeros(1, 3)
    assert torch.allclose(V.gaussian_kl_closed_form(mu, logvar), torch.zeros(1))  # q = p
    mu = torch.tensor([[1.0]]); logvar = torch.tensor([[0.0]])
    assert torch.isclose(V.gaussian_kl_closed_form(mu, logvar)[0], torch.tensor(0.5))  # ½ mu²
    mu = torch.tensor([[0.0]]); logvar = torch.tensor([[math.log(4.0)]])  # sigma² = 4
    expected = 0.5 * (4.0 - math.log(4.0) - 1.0)
    assert torch.isclose(V.gaussian_kl_closed_form(mu, logvar)[0], torch.tensor(expected))


def test_gaussian_kl_closed_form_matches_monte_carlo():
    mu = torch.randn(4, 3); logvar = 0.5 * torch.randn(4, 3)
    closed = V.gaussian_kl_closed_form(mu, logvar)
    mc = V.gaussian_kl_monte_carlo(mu, logvar, n_samples=20000)
    assert torch.allclose(closed, mc, atol=0.05, rtol=0.05)


def test_gaussian_log_density_matches_torch_distributions():
    z, mu, logvar = torch.randn(6, 3), torch.randn(6, 3), torch.randn(6, 3)
    ref = torch.distributions.Normal(mu, (0.5 * logvar).exp()).log_prob(z).sum(-1)
    assert torch.allclose(V.gaussian_log_density(z, mu, logvar), ref, atol=1e-5)


def test_vae_loss_components():
    x = torch.randn(8, 2); x_hat = x.clone(); mu = torch.zeros(8, 4); logvar = torch.zeros(8, 4)
    const = 2 * 0.5 * math.log(2 * math.pi)  # d_x · ½ log 2π with sigma_dec = 1
    loss, recon, kl = V.vae_loss(x, x_hat, mu, logvar, beta=1.0)
    assert torch.isclose(recon, torch.tensor(const)) and kl.item() == 0.0
    assert torch.isclose(loss, torch.tensor(const))
    loss_b, _, _ = V.vae_loss(x, x_hat + 1.0, torch.ones(8, 4), logvar, beta=2.0)
    # recon = ½·2 + const = 1 + const ; kl = ½·4·1 = 2 ; loss = 1 + const + 2·2 = 5 + const
    assert torch.isclose(loss_b, torch.tensor(5.0 + const))


def test_negative_elbo_estimate_agrees_with_vae_loss():
    torch.manual_seed(1)
    model = V.VAE(d_x=2, d_z=2, d_hidden=16)
    x = torch.randn(256, 2)
    est = torch.stack([V.negative_elbo_estimate(model, x, sigma_dec=0.5) for _ in range(20)]).mean()
    direct = torch.stack([V.vae_loss(x, model(x)[0], *model.encode(x), sigma_dec=0.5)[0] for _ in range(20)]).mean()
    assert torch.isclose(est, direct, rtol=0.05)


def test_decoder_sigma_is_the_rate_distortion_knob():
    """sigma_dec trades rate (KL, nats through the bottleneck) against distortion (MSE).

    Data is 4 modes on a radius-2 circle: naming the mode costs log 4 = 1.39 nats, so a model
    whose KL is below that cannot even be encoding which mode the point came from.
    """
    x = gaussian_mixture_2d(512, n_modes=4, std=0.2)
    kl_by_sigma, mse_by_sigma = {}, {}
    for sigma in (1.0, 0.3, 0.1):
        torch.manual_seed(0)
        model = V.VAE(d_x=2, d_z=2, d_hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(400):
            x_hat, mu, logvar = model(x)
            loss, recon, kl = V.vae_loss(x, x_hat, mu, logvar, sigma_dec=sigma)
            opt.zero_grad(); loss.backward(); opt.step()
        kl_by_sigma[sigma] = kl.item()
        mse_by_sigma[sigma] = ((x - model(x)[0]) ** 2).sum(dim=1).mean().item()
    assert kl_by_sigma[1.0] < kl_by_sigma[0.3] < kl_by_sigma[0.1]     # rate rises as sigma falls
    assert mse_by_sigma[1.0] > mse_by_sigma[0.3] > mse_by_sigma[0.1]  # distortion falls
    assert kl_by_sigma[1.0] < math.log(4.0)   # sigma = 1: not even the mode index gets through
    assert mse_by_sigma[0.1] < 0.1            # sigma = 0.1: the latent is genuinely used


def test_vae_negative_elbo_decreases_on_synthetic_data():
    x = gaussian_mixture_2d(512, n_modes=4, std=0.2)
    model = V.VAE(d_x=2, d_z=2, d_hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    x_hat, mu, logvar = model(x)
    first = V.vae_loss(x, x_hat, mu, logvar, sigma_dec=0.1)[0].item()
    for _ in range(500):
        x_hat, mu, logvar = model(x)
        loss, recon, kl = V.vae_loss(x, x_hat, mu, logvar, sigma_dec=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2 * first
    assert recon.item() < 0.3 * first   # reconstruction, not the KL, did the work
    assert torch.isfinite(V.negative_elbo_estimate(model, x, sigma_dec=0.1))


# ---------------------------------------------------------------- vqvae.py
def test_vector_quantizer_nearest_code_and_straight_through():
    vq = VQ.VectorQuantizer(n_codes=4, d_code=2, beta=0.25)
    with torch.no_grad():
        vq.codebook.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    z_e = torch.tensor([[[0.9, 0.1], [0.1, 0.9], [0.4, 0.6]]], requires_grad=True)  # (1, 3, 2)
    z_q, vq_loss, idx = vq(z_e)
    assert idx.tolist() == [[1, 2, 2]]
    assert torch.allclose(z_q.detach(), vq.codebook.weight[idx])  # forward value is the code
    # straight-through: d(sum z_q)/d z_e == 1 everywhere
    z_q.sum().backward()
    assert torch.allclose(z_e.grad, torch.ones_like(z_e))
    assert vq.codebook.weight.grad is None or torch.all(vq.codebook.weight.grad == 0)


def test_vq_loss_pulls_codebook_and_encoder_correctly():
    vq = VQ.VectorQuantizer(n_codes=2, d_code=2, beta=0.25)
    with torch.no_grad():
        vq.codebook.weight.copy_(torch.tensor([[0.0, 0.0], [5.0, 5.0]]))
    z_e = torch.tensor([[[1.0, 1.0]]], requires_grad=True)
    _, vq_loss, _ = vq(z_e)
    # codebook term ||sg[z_e] − e||² = mean(1,1) = 1 ; commitment 0.25·1 → 1.25
    assert torch.isclose(vq_loss, torch.tensor(1.25))
    vq_loss.backward()
    # codebook term moves e_0 toward z_e; commitment term moves z_e toward e_0 (scaled by beta)
    assert torch.all(vq.codebook.weight.grad[0] < 0)  # gradient of (z−e)² wrt e is −2(z−e)/D < 0
    assert torch.all(z_e.grad > 0)


def test_codebook_perplexity_bounds():
    assert torch.isclose(VQ.codebook_perplexity(torch.zeros(1, 10, dtype=torch.long), 8), torch.tensor(1.0))
    assert torch.isclose(VQ.codebook_perplexity(torch.arange(8)[None], 8), torch.tensor(8.0))


def test_vqvae_trains():
    x = gaussian_mixture_2d(256, n_modes=4, std=0.1)
    model = VQ.VQVAE(d_x=2, d_code=4, n_codes=8)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    x_hat, vq_loss, _ = model(x)
    first = VQ.vqvae_loss(x, x_hat, vq_loss).item()
    for _ in range(300):
        x_hat, vq_loss, idx = model(x)
        loss = VQ.vqvae_loss(x, x_hat, vq_loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first
    assert VQ.codebook_perplexity(idx[None], 8) > 1.5  # more than one code in use
