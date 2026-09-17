# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/generative/gan.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gan -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py generative/gan --force

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

    def __init__(self, d_z: int, d_x: int, d_hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class Discriminator(nn.Module):
    """``x (B, d_x) -> logit (B,)``;  D(x) = sigmoid(logit)."""

    def __init__(self, d_x: int, d_hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def d_loss_standard(logit_real: torch.Tensor, logit_fake: torch.Tensor) -> torch.Tensor:
    """``−E[log D(x)] − E[log(1 − D(G(z)))]``: the discriminator *minimises* this.

    Args:
        logit_real, logit_fake: (B,).
    Returns:
        scalar.  At the optimum D* with p_g = p_data this equals log 4 ≈ 1.386.
    """
    raise NotImplementedError('TODO: implement d_loss_standard (see the reference in src/mlbook)')

def g_loss_saturating(logit_fake: torch.Tensor) -> torch.Tensor:
    """Minimax generator loss ``E[log(1 − D(G(z)))]``.  Its gradient vanishes when D is confident."""
    raise NotImplementedError('TODO: implement g_loss_saturating (see the reference in src/mlbook)')

def g_loss_non_saturating(logit_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss ``−E[log D(G(z))]`` (the one everyone actually trains with).

    Args:
        logit_fake: (B,).
    Returns:
        scalar.
    """
    raise NotImplementedError('TODO: implement g_loss_non_saturating (see the reference in src/mlbook)')

def d_loss_wasserstein(critic_real: torch.Tensor, critic_fake: torch.Tensor) -> torch.Tensor:
    """WGAN critic loss ``−(E[f(x)] − E[f(G(z))])`` for a 1-Lipschitz critic ``f``.

    Args:
        critic_real, critic_fake: (B,) raw critic values (no sigmoid).
    """
    raise NotImplementedError('TODO: implement d_loss_wasserstein (see the reference in src/mlbook)')

def g_loss_wasserstein(critic_fake: torch.Tensor) -> torch.Tensor:
    """WGAN generator loss ``−E[f(G(z))]``."""
    raise NotImplementedError('TODO: implement g_loss_wasserstein (see the reference in src/mlbook)')

def gradient_penalty(critic: nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    """WGAN-GP penalty ``E_{x̂}[(||∇_{x̂} f(x̂)||₂ − 1)²]`` on random interpolates x̂.

    Args:
        x_real, x_fake: (B, d_x).
    Returns:
        scalar.
    """
    raise NotImplementedError('TODO: implement gradient_penalty (see the reference in src/mlbook)')

def gan_train_step(gen: Generator, disc: Discriminator, opt_g: torch.optim.Optimizer, opt_d: torch.optim.Optimizer, x_real: torch.Tensor, d_z: int) -> tuple[float, float]:
    """One alternating update: D step on (real, detached fake), then G step (non-saturating).

    Args:
        x_real: (B, d_x).
    Returns:
        (d_loss, g_loss) as floats.
    """
    raise NotImplementedError('TODO: implement gan_train_step (see the reference in src/mlbook)')

def spectral_norm_power_iteration(w: torch.Tensor, u: torch.Tensor, n_iter: int=1) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest singular value of ``w`` by power iteration (what ``nn.utils.spectral_norm`` does).

    Args:
        w: (d_out, d_in).  u: (d_out,) persistent left singular vector estimate.
    Returns:
        (sigma scalar, u_new (d_out,)).  Divide ``w`` by ``sigma`` to make the layer 1-Lipschitz.
    """
    raise NotImplementedError('TODO: implement spectral_norm_power_iteration (see the reference in src/mlbook)')
