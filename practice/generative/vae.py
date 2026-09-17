# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/generative/vae.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k vae -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py generative/vae --force

"""Variational autoencoder with a diagonal-Gaussian encoder and a Gaussian decoder.

The objective is the negative ELBO

    -ELBO(x) = E_{q(z|x)}[ -log p(x|z) ] + beta * KL( q(z|x) || N(0, I) ),

with ``q(z|x) = N(mu(x), diag(sigma(x)²))`` and, for a unit-variance Gaussian decoder,
``-log p(x|z) = ½ ||x - mu_dec(z)||² + const``.  ``beta = 1`` is the true ELBO; ``beta > 1``
is the beta-VAE trade-off.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from mlbook.generative.autoencoder import mlp

class VAE(nn.Module):
    """MLP VAE.  encode: (B, d_x) -> (mu, logvar) each (B, d_z);  decode: (B, d_z) -> (B, d_x)."""

    def __init__(self, d_x: int, d_z: int, d_hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``z = mu + sigma * eps``, ``eps ~ N(0, I)``: a sample of q(z|x) that is differentiable in mu, sigma.

    Args:
        mu, logvar: (B, d_z).
    Returns:
        z: (B, d_z).
    """
    raise NotImplementedError('TODO: implement reparameterize (see the reference in src/mlbook)')

def gaussian_kl_closed_form(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``KL( N(mu, diag(sigma²)) || N(0, I) ) = ½ Σ_j ( mu_j² + sigma_j² − log sigma_j² − 1 )``.

    Args:
        mu, logvar: (B, d_z).
    Returns:
        kl: (B,)  one KL per example (summed over latent dimensions).
    """
    raise NotImplementedError('TODO: implement gaussian_kl_closed_form (see the reference in src/mlbook)')

def gaussian_kl_monte_carlo(mu: torch.Tensor, logvar: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Monte-Carlo estimate ``E_q[ log q(z) − log p(z) ]`` with ``z ~ q`` (for checking the closed form).

    Args:
        mu, logvar: (B, d_z).
    Returns:
        kl: (B,).
    """
    raise NotImplementedError('TODO: implement gaussian_kl_monte_carlo (see the reference in src/mlbook)')

def gaussian_log_density(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``log N(z; mu, diag(exp(logvar)))`` summed over the last dimension.

    Args:
        z, mu, logvar: broadcastable to (..., d).
    Returns:
        (...,) log-density.
    """
    raise NotImplementedError('TODO: implement gaussian_log_density (see the reference in src/mlbook)')

def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=1.0, sigma_dec: float=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative (beta-)ELBO per example, averaged over the batch.

    recon = −log p(x|z) = ||x − x_hat||² / (2 σ²) + d_x (log σ + ½ log 2π)
    kl    = KL(q(z|x) || N(0, I))   (closed form)
    loss  = recon + beta · kl

    ``sigma_dec`` is the fixed standard deviation of the Gaussian decoder and it *is* the
    rate–distortion knob: 1/σ² multiplies the reconstruction term, so a large σ on data of
    scale ≫ σ makes ignoring z cheaper than encoding it, which is posterior collapse.  Reporting a
    β-VAE with β = 1 and σ = 1 on unnormalised data is the classic way to get a blurry mean.

    Args:
        x, x_hat: (B, d_x).  mu, logvar: (B, d_z).
    Returns:
        (loss, recon, kl) scalars.
    """
    raise NotImplementedError('TODO: implement vae_loss (see the reference in src/mlbook)')

def negative_elbo_estimate(model: VAE, x: torch.Tensor, sigma_dec: float=1.0) -> torch.Tensor:
    """Single-sample estimate of −ELBO(x) built from ``gaussian_log_density`` (for monitoring).

    Equals ``vae_loss(..., beta=1, sigma_dec=sigma_dec)`` in expectation, a useful cross-check
    that the closed-form KL and the hand-written Gaussian NLL agree.

    Args:
        x: (B, d_x).
    Returns:
        scalar, mean over the batch.
    """
    raise NotImplementedError('TODO: implement negative_elbo_estimate (see the reference in src/mlbook)')
