"""A minimal GAN on vectors: generator, discriminator, and the standard / non-saturating /
Wasserstein losses, plus the gradient penalty.

Minimax game:  min_G max_D  E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 − D(G(z)))].
The discriminator outputs a *logit*; ``sigmoid(logit) = D(x)``.  Using
``binary_cross_entropy_with_logits`` keeps ``log(1 − sigmoid(l)) = −softplus(l)`` numerically safe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mlbook.generative.autoencoder import mlp


class Generator(nn.Module):
    """``z (B, d_z) -> x_fake (B, d_x)``."""

    def __init__(self, d_z: int, d_x: int, d_hidden: int = 64) -> None:
        super().__init__()
        self.net = mlp(d_z, d_hidden, d_x)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_fake = self.net(z)                                   # (B, d_x)
        return x_fake


class Discriminator(nn.Module):
    """``x (B, d_x) -> logit (B,)``;  D(x) = sigmoid(logit)."""

    def __init__(self, d_x: int, d_hidden: int = 64) -> None:
        super().__init__()
        self.net = mlp(d_x, d_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.net(x)[:, 0]                              # (B,)
        return logit


def d_loss_standard(logit_real: torch.Tensor, logit_fake: torch.Tensor) -> torch.Tensor:
    """``−E[log D(x)] − E[log(1 − D(G(z)))]``: the discriminator *minimises* this.

    Args:
        logit_real, logit_fake: (B,).
    Returns:
        scalar.  At the optimum D* with p_g = p_data this equals log 4 ≈ 1.386.
    """
    loss_real = F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
    loss_fake = F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))
    return loss_real + loss_fake


def g_loss_saturating(logit_fake: torch.Tensor) -> torch.Tensor:
    """Minimax generator loss ``E[log(1 − D(G(z)))]``.  Its gradient vanishes when D is confident."""
    return -F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))


def g_loss_non_saturating(logit_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss ``−E[log D(G(z))]`` (the one everyone actually trains with).

    Args:
        logit_fake: (B,).
    Returns:
        scalar.
    """
    return F.binary_cross_entropy_with_logits(logit_fake, torch.ones_like(logit_fake))


def d_loss_wasserstein(critic_real: torch.Tensor, critic_fake: torch.Tensor) -> torch.Tensor:
    """WGAN critic loss ``−(E[f(x)] − E[f(G(z))])`` for a 1-Lipschitz critic ``f``.

    Args:
        critic_real, critic_fake: (B,) raw critic values (no sigmoid).
    """
    return -(critic_real.mean() - critic_fake.mean())


def g_loss_wasserstein(critic_fake: torch.Tensor) -> torch.Tensor:
    """WGAN generator loss ``−E[f(G(z))]``."""
    return -critic_fake.mean()


def gradient_penalty(critic: nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    """WGAN-GP penalty ``E_{x̂}[(||∇_{x̂} f(x̂)||₂ − 1)²]`` on random interpolates x̂.

    Args:
        x_real, x_fake: (B, d_x).
    Returns:
        scalar.
    """
    alpha = torch.rand(x_real.shape[0], 1)                                    # (B, 1)
    x_hat = alpha * x_real + (1.0 - alpha) * x_fake                           # (B, d_x)
    x_hat.requires_grad_(True)
    f = critic(x_hat)                                                         # (B,)
    grad = torch.autograd.grad(f.sum(), x_hat, create_graph=True)[0]          # (B, d_x)
    grad_norm = grad.norm(dim=1)                                              # (B,)
    return ((grad_norm - 1.0) ** 2).mean()


def gan_train_step(gen: Generator, disc: Discriminator, opt_g: torch.optim.Optimizer,
                   opt_d: torch.optim.Optimizer, x_real: torch.Tensor, d_z: int) -> tuple[float, float]:
    """One alternating update: D step on (real, detached fake), then G step (non-saturating).

    Args:
        x_real: (B, d_x).
    Returns:
        (d_loss, g_loss) as floats.
    """
    B = x_real.shape[0]
    # --- discriminator step -------------------------------------------------
    z = torch.randn(B, d_z)                                                   # (B, d_z)
    x_fake = gen(z).detach()                                                  # (B, d_x)  no grad into G
    d_loss = d_loss_standard(disc(x_real), disc(x_fake))
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()
    # --- generator step -----------------------------------------------------
    z = torch.randn(B, d_z)                                                   # (B, d_z)
    g_loss = g_loss_non_saturating(disc(gen(z)))                              # grads flow through D into G
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()
    return float(d_loss), float(g_loss)


def spectral_norm_power_iteration(w: torch.Tensor, u: torch.Tensor, n_iter: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest singular value of ``w`` by power iteration (what ``nn.utils.spectral_norm`` does).

    Args:
        w: (d_out, d_in).  u: (d_out,) persistent left singular vector estimate.
    Returns:
        (sigma scalar, u_new (d_out,)).  Divide ``w`` by ``sigma`` to make the layer 1-Lipschitz.
    """
    for _ in range(n_iter):
        v = F.normalize(w.t() @ u, dim=0)                                     # (d_in,)
        u = F.normalize(w @ v, dim=0)                                         # (d_out,)
    sigma = u @ (w @ v)                                                       # scalar  = u^T W v
    return sigma, u
