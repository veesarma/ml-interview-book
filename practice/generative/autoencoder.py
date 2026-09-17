# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/generative/autoencoder.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k autoencoder -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py generative/autoencoder --force

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
    raise NotImplementedError('TODO: implement mlp (see the reference in src/mlbook)')

class Autoencoder(nn.Module):
    """Deterministic bottleneck autoencoder.

    encode: (B, d_x) -> (B, d_z);  decode: (B, d_z) -> (B, d_x).
    Loss: ``||x - decode(encode(x))||²`` averaged over the batch.
    """

    def __init__(self, d_x: int, d_z: int, d_hidden: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of the per-example squared error ``sum_j (x_j - x_hat_j)²``.

    Args:
        x, x_hat: (B, d_x).
    Returns:
        scalar.
    """
    raise NotImplementedError('TODO: implement reconstruction_loss (see the reference in src/mlbook)')

def corrupt_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Denoising-AE corruption ``x_tilde = x + sigma * eps``,  eps ~ N(0, I).

    Args:
        x: (B, d_x).
    Returns:
        x_tilde: (B, d_x).
    """
    raise NotImplementedError('TODO: implement corrupt_gaussian (see the reference in src/mlbook)')

def corrupt_mask(x: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Masking corruption: each coordinate is zeroed independently with probability ``p_drop``.

    Args:
        x: (B, d_x).
    Returns:
        x_tilde: (B, d_x).
    """
    raise NotImplementedError('TODO: implement corrupt_mask (see the reference in src/mlbook)')

def denoising_loss(model: Autoencoder, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Denoising objective: reconstruct clean ``x`` from ``x + sigma * eps``.

    Args:
        x: (B, d_x).
    Returns:
        scalar loss.
    """
    raise NotImplementedError('TODO: implement denoising_loss (see the reference in src/mlbook)')
