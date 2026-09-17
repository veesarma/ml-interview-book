"""Plain and denoising autoencoders (MLP encoder / decoder).

An autoencoder learns ``x -> z -> x_hat`` with a bottleneck ``d_z < d_x`` so that ``z`` must
keep the information that matters for reconstruction.  A *denoising* autoencoder is the same
network trained to reconstruct the clean ``x`` from a corrupted ``x + noise``; this is the
ancestor of both masked autoencoders (Part X) and diffusion models (Part IX, chapter 3).
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    """Two-hidden-layer MLP ``d_in -> d_hidden -> d_hidden -> d_out`` with SiLU activations."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, d_out),
    )


class Autoencoder(nn.Module):
    """Deterministic bottleneck autoencoder.

    encode: (B, d_x) -> (B, d_z);  decode: (B, d_z) -> (B, d_x).
    Loss: ``||x - decode(encode(x))||²`` averaged over the batch.
    """

    def __init__(self, d_x: int, d_z: int, d_hidden: int = 64) -> None:
        super().__init__()
        self.encoder = mlp(d_x, d_hidden, d_z)
        self.decoder = mlp(d_z, d_hidden, d_x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)                                   # (B, d_z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)                               # (B, d_x)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)                                    # (B, d_z)
        x_hat = self.decode(z)                                # (B, d_x)
        return x_hat


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of the per-example squared error ``sum_j (x_j - x_hat_j)²``.

    Args:
        x, x_hat: (B, d_x).
    Returns:
        scalar.
    """
    per_example = ((x - x_hat) ** 2).sum(dim=1)              # (B,)
    return per_example.mean()


def corrupt_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Denoising-AE corruption ``x_tilde = x + sigma * eps``,  eps ~ N(0, I).

    Args:
        x: (B, d_x).
    Returns:
        x_tilde: (B, d_x).
    """
    eps = torch.randn_like(x)                                 # (B, d_x)
    return x + sigma * eps


def corrupt_mask(x: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Masking corruption: each coordinate is zeroed independently with probability ``p_drop``.

    Args:
        x: (B, d_x).
    Returns:
        x_tilde: (B, d_x).
    """
    keep = (torch.rand_like(x) >= p_drop).to(x.dtype)         # (B, d_x) in {0, 1}
    return x * keep


def denoising_loss(model: Autoencoder, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Denoising objective: reconstruct clean ``x`` from ``x + sigma * eps``.

    Args:
        x: (B, d_x).
    Returns:
        scalar loss.
    """
    x_tilde = corrupt_gaussian(x, sigma)                      # (B, d_x)
    x_hat = model(x_tilde)                                    # (B, d_x)
    return reconstruction_loss(x, x_hat)
