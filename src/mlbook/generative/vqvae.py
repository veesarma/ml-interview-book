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

    def __init__(self, n_codes: int, d_code: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook = nn.Embedding(n_codes, d_code)                  # weight: (K, D)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)
        self.beta = beta

    def nearest_code(self, z_e: torch.Tensor) -> torch.Tensor:
        """Index of the closest codebook vector for every input vector.

        Args:
            z_e: (B, N, D).
        Returns:
            indices: (B, N) integers in [0, K).
        """
        flat = z_e.reshape(-1, z_e.shape[-1])                          # (B*N, D)
        e = self.codebook.weight                                       # (K, D)
        # ||z − e||² = ||z||² − 2 z·e + ||e||²   (expanded so we never build a (B*N, K, D) tensor)
        d2 = (flat ** 2).sum(dim=1, keepdim=True) - 2.0 * flat @ e.t() + (e ** 2).sum(dim=1)[None, :]  # (B*N, K)
        indices = d2.argmin(dim=1)                                     # (B*N,)
        return indices.reshape(z_e.shape[0], z_e.shape[1])             # (B, N)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.nearest_code(z_e)                               # (B, N)
        z_q = self.codebook(indices)                                   # (B, N, D)
        codebook_loss = ((z_e.detach() - z_q) ** 2).mean()             # pulls e_k toward z_e
        commitment_loss = ((z_e - z_q.detach()) ** 2).mean()           # pulls z_e toward e_k
        vq_loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()                            # (B, N, D) straight-through
        return z_q_st, vq_loss, indices


def codebook_perplexity(indices: torch.Tensor, n_codes: int) -> torch.Tensor:
    """``exp(H(usage))`` of the code histogram: how many codes are effectively in use.

    Args:
        indices: (B, N) integers in [0, K).
    Returns:
        scalar in [1, K].
    """
    counts = torch.bincount(indices.reshape(-1), minlength=n_codes).float()  # (K,)
    p = counts / counts.sum()                                                 # (K,)
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return torch.exp(entropy)


class VQVAE(nn.Module):
    """MLP VQ-VAE for vectors: encode -> quantise -> decode.

    encode: (B, d_x) -> z_e (B, 1, D);  decode: z_q (B, 1, D) -> (B, d_x).
    """

    def __init__(self, d_x: int, d_code: int, n_codes: int, d_hidden: int = 64, beta: float = 0.25) -> None:
        super().__init__()
        from mlbook.generative.autoencoder import mlp

        self.encoder = mlp(d_x, d_hidden, d_code)
        self.quantizer = VectorQuantizer(n_codes, d_code, beta)
        self.decoder = mlp(d_code, d_hidden, d_x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)[:, None, :]                              # (B, 1, D)
        z_q, vq_loss, indices = self.quantizer(z_e)                    # (B, 1, D), scalar, (B, 1)
        x_hat = self.decoder(z_q[:, 0, :])                             # (B, d_x)
        return x_hat, vq_loss, indices[:, 0]                           # indices: (B,)


def vqvae_loss(x: torch.Tensor, x_hat: torch.Tensor, vq_loss: torch.Tensor) -> torch.Tensor:
    """``||x − x_hat||² (mean over batch, summed over dims) + vq_loss``."""
    recon = ((x - x_hat) ** 2).sum(dim=1).mean()
    return recon + vq_loss
