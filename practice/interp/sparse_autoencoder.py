# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/interp/sparse_autoencoder.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k sparse_autoencoder -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py interp/sparse_autoencoder --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        raise NotImplementedError('TODO: implement normalize_decoder (see the reference in src/mlbook)')

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def sae_loss(x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor, l1_coeff: float) -> torch.Tensor:
    """Mean over batch of ||x - x_hat||^2 + lambda ||f||_1."""
    raise NotImplementedError('TODO: implement sae_loss (see the reference in src/mlbook)')

def make_superposition_data(n: int, d: int, n_true: int, p_active: float, seed: int=0) -> tuple[torch.Tensor, torch.Tensor]:
    """n activations that are sparse combinations of n_true > d random unit directions.

    Returns (X (n, d), directions (n_true, d)).
    """
    raise NotImplementedError('TODO: implement make_superposition_data (see the reference in src/mlbook)')

def train_sae(sae: SparseAutoencoder, X: torch.Tensor, l1_coeff: float, steps: int, lr: float=0.01, batch: int=256) -> float:
    """Adam on sae_loss with decoder re-normalisation after each step; returns final loss."""
    raise NotImplementedError('TODO: implement train_sae (see the reference in src/mlbook)')

def feature_recovery(sae: SparseAutoencoder, dirs: torch.Tensor) -> torch.Tensor:
    """For each true direction, the best |cosine| with any learned decoder row: (n_true,)."""
    raise NotImplementedError('TODO: implement feature_recovery (see the reference in src/mlbook)')
