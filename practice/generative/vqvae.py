# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/generative/vqvae.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k vqvae -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py generative/vqvae --force

"""Vector-quantised bottleneck (VQ-VAE) with the straight-through estimator.

Given encoder outputs ``z_e`` and a codebook ``e ∈ R^{K×D}``, quantisation picks the nearest
code ``z_q = e_k, k = argmin_j ||z_e − e_j||``.  The loss is

    L = ||x − decode(z_q)||²  +  ||sg[z_e] − e_k||²  +  beta · ||z_e − sg[e_k]||²
        reconstruction           codebook (moves e_k to z_e)     commitment (keeps z_e near e_k)

and gradients cross the non-differentiable argmin with the straight-through trick
``z_q = z_e + sg[z_q − z_e]`` (forward: z_q; backward: identity to z_e).
"""
from __future__ import annotations
import torch
from torch import nn

class VectorQuantizer(nn.Module):
    """Nearest-neighbour quantiser with codebook (K, D).

    forward: z_e (B, N, D) -> (z_q (B, N, D), vq_loss scalar, indices (B, N))
    """

    def __init__(self, n_codes: int, d_code: int, beta: float=0.25) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def nearest_code(self, z_e: torch.Tensor) -> torch.Tensor:
        """Index of the closest codebook vector for every input vector.

        Args:
            z_e: (B, N, D).
        Returns:
            indices: (B, N) integers in [0, K).
        """
        raise NotImplementedError('TODO: implement nearest_code (see the reference in src/mlbook)')

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def codebook_perplexity(indices: torch.Tensor, n_codes: int) -> torch.Tensor:
    """``exp(H(usage))`` of the code histogram: how many codes are effectively in use.

    Args:
        indices: (B, N) integers in [0, K).
    Returns:
        scalar in [1, K].
    """
    raise NotImplementedError('TODO: implement codebook_perplexity (see the reference in src/mlbook)')

class VQVAE(nn.Module):
    """MLP VQ-VAE for vectors: encode -> quantise -> decode.

    encode: (B, d_x) -> z_e (B, 1, D);  decode: z_q (B, 1, D) -> (B, d_x).
    """

    def __init__(self, d_x: int, d_code: int, n_codes: int, d_hidden: int=64, beta: float=0.25) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def vqvae_loss(x: torch.Tensor, x_hat: torch.Tensor, vq_loss: torch.Tensor) -> torch.Tensor:
    """``||x − x_hat||² (mean over batch, summed over dims) + vq_loss``."""
    raise NotImplementedError('TODO: implement vqvae_loss (see the reference in src/mlbook)')
