"""A sparse autoencoder (SAE) for dictionary learning on activations
(Bricken et al. 2023, "Towards Monosemanticity").

    f(x) = ReLU( (x - b_dec) W_enc + b_enc )        features, (B, F), F > d
    x_hat = f(x) W_dec + b_dec                         reconstruction, (B, d)
    L = ||x - x_hat||^2_2 + lambda * ||f(x)||_1        with rows of W_dec unit-norm

The L1 penalty makes most features zero on any given input; unit-norm decoder rows
stop the model from shrinking f and growing W_dec to cheat the penalty.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SparseAutoencoder(nn.Module):
    """x (B, d) -> features (B, F) -> x_hat (B, d)."""

    def __init__(self, d: int, n_features: int):
        super().__init__()
        self.W_dec = nn.Parameter(torch.randn(n_features, d))  # (F, d)
        self.b_dec = nn.Parameter(torch.zeros(d))  # (d,)
        self.normalize_decoder()
        # Tied initialisation W_enc = W_dec^T: each feature starts by detecting its own direction.
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone())  # (d, F)
        self.b_enc = nn.Parameter(torch.zeros(n_features))  # (F,)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)  # unit-norm rows (F, d)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)  # (B, F)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec  # (B, d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)  # (B, F)
        return self.decode(f), f  # (B, d), (B, F)


def sae_loss(x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor, l1_coeff: float) -> torch.Tensor:
    """Mean over batch of ||x - x_hat||^2 + lambda ||f||_1."""
    recon = ((x - x_hat) ** 2).sum(dim=1).mean()  # scalar
    sparsity = f.abs().sum(dim=1).mean()  # scalar
    return recon + l1_coeff * sparsity


def make_superposition_data(n: int, d: int, n_true: int, p_active: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """n activations that are sparse combinations of n_true > d random unit directions.

    Returns (X (n, d), directions (n_true, d)).
    """
    g = torch.Generator().manual_seed(seed)
    dirs = F.normalize(torch.randn(n_true, d, generator=g), dim=1)  # (n_true, d)
    active = (torch.rand(n, n_true, generator=g) < p_active).float()  # (n, n_true)
    mags = torch.rand(n, n_true, generator=g) + 0.5  # (n, n_true)
    return (active * mags) @ dirs, dirs  # (n, d)


def train_sae(sae: SparseAutoencoder, X: torch.Tensor, l1_coeff: float, steps: int, lr: float = 1e-2, batch: int = 256) -> float:
    """Adam on sae_loss with decoder re-normalisation after each step; returns final loss."""
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (batch,), generator=g)  # (batch,)
        x = X[idx]  # (batch, d)
        x_hat, f = sae(x)
        loss = sae_loss(x, x_hat, f, l1_coeff)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
    return float(loss.detach())


def feature_recovery(sae: SparseAutoencoder, dirs: torch.Tensor) -> torch.Tensor:
    """For each true direction, the best |cosine| with any learned decoder row: (n_true,)."""
    cos = dirs @ sae.W_dec.detach().T  # (n_true, F)
    return cos.abs().amax(dim=1)  # (n_true,)
