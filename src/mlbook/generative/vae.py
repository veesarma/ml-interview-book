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

    def __init__(self, d_x: int, d_z: int, d_hidden: int = 64) -> None:
        super().__init__()
        self.encoder_trunk = mlp(d_x, d_hidden, d_hidden)
        self.to_mu = nn.Linear(d_hidden, d_z)      # separate heads, never a fused 2*d_z projection
        self.to_logvar = nn.Linear(d_hidden, d_z)
        self.decoder = mlp(d_z, d_hidden, d_x)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(x)                              # (B, d_hidden)
        mu = self.to_mu(h)                                     # (B, d_z)
        logvar = self.to_logvar(h)                             # (B, d_z)  log sigma²
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)                                # (B, d_x)  = mean of p(x|z)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)                            # (B, d_z), (B, d_z)
        z = reparameterize(mu, logvar)                         # (B, d_z)
        x_hat = self.decode(z)                                 # (B, d_x)
        return x_hat, mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``z = mu + sigma * eps``, ``eps ~ N(0, I)``: a sample of q(z|x) that is differentiable in mu, sigma.

    Args:
        mu, logvar: (B, d_z).
    Returns:
        z: (B, d_z).
    """
    sigma = torch.exp(0.5 * logvar)                            # (B, d_z)
    eps = torch.randn_like(mu)                                 # (B, d_z)
    z = mu + sigma * eps                                       # (B, d_z)
    return z


def gaussian_kl_closed_form(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``KL( N(mu, diag(sigma²)) || N(0, I) ) = ½ Σ_j ( mu_j² + sigma_j² − log sigma_j² − 1 )``.

    Args:
        mu, logvar: (B, d_z).
    Returns:
        kl: (B,)  one KL per example (summed over latent dimensions).
    """
    per_dim = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1.0)   # (B, d_z)
    kl = per_dim.sum(dim=1)                                    # (B,)
    return kl


def gaussian_kl_monte_carlo(mu: torch.Tensor, logvar: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Monte-Carlo estimate ``E_q[ log q(z) − log p(z) ]`` with ``z ~ q`` (for checking the closed form).

    Args:
        mu, logvar: (B, d_z).
    Returns:
        kl: (B,).
    """
    sigma = torch.exp(0.5 * logvar)                            # (B, d_z)
    eps = torch.randn(n_samples, *mu.shape)                    # (S, B, d_z)
    z = mu[None] + sigma[None] * eps                           # (S, B, d_z)
    log_q = gaussian_log_density(z, mu[None], logvar[None])    # (S, B)
    log_p = gaussian_log_density(z, torch.zeros_like(z), torch.zeros_like(z))  # (S, B)
    kl = (log_q - log_p).mean(dim=0)                           # (B,)
    return kl


def gaussian_log_density(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``log N(z; mu, diag(exp(logvar)))`` summed over the last dimension.

    Args:
        z, mu, logvar: broadcastable to (..., d).
    Returns:
        (...,) log-density.
    """
    per_dim = -0.5 * (math.log(2 * math.pi) + logvar + (z - mu) ** 2 / logvar.exp())  # (..., d)
    return per_dim.sum(dim=-1)


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative (beta-)ELBO per example, averaged over the batch.

    recon = ½ ||x − x_hat||²   (unit-variance Gaussian decoder, constants dropped)
    kl    = KL(q(z|x) || N(0, I))   (closed form)
    loss  = recon + beta · kl

    Args:
        x, x_hat: (B, d_x).  mu, logvar: (B, d_z).
    Returns:
        (loss, recon, kl) scalars.
    """
    recon = 0.5 * ((x - x_hat) ** 2).sum(dim=1)                # (B,)
    kl = gaussian_kl_closed_form(mu, logvar)                   # (B,)
    loss = (recon + beta * kl).mean()
    return loss, recon.mean(), kl.mean()


def negative_elbo_estimate(model: VAE, x: torch.Tensor) -> torch.Tensor:
    """Single-sample estimate of −ELBO(x) including the Gaussian constants (for monitoring).

    Args:
        x: (B, d_x).
    Returns:
        scalar, mean over the batch.
    """
    mu, logvar = model.encode(x)                               # (B, d_z), (B, d_z)
    z = reparameterize(mu, logvar)                             # (B, d_z)
    x_hat = model.decode(z)                                    # (B, d_x)
    log_px_given_z = gaussian_log_density(x, x_hat, torch.zeros_like(x))  # (B,)  unit variance
    kl = gaussian_kl_closed_form(mu, logvar)                   # (B,)
    return (-log_px_given_z + kl).mean()
