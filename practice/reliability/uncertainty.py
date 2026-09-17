# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/reliability/uncertainty.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k uncertainty -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py reliability/uncertainty --force

"""Uncertainty: heteroscedastic Gaussian NLL, predictive entropy, mutual
information (BALD), MC dropout and deep-ensemble prediction helpers.

Given M stochastic forward passes p_m = p(y | x, theta_m), m = 1..M, with mean
p_bar = (1/M) sum_m p_m:
    total uncertainty      H[p_bar]
    aleatoric (expected)   (1/M) sum_m H[p_m]
    epistemic (MI / BALD)  H[p_bar] - (1/M) sum_m H[p_m]   >= 0
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

def heteroscedastic_gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example NLL of N(y | mu, sigma^2) up to a constant:
        0.5 * log sigma^2 + (y - mu)^2 / (2 sigma^2).
    mu, log_var, y all (N,) or (N, 1). Returns the mean over examples."""
    raise NotImplementedError('TODO: implement heteroscedastic_gaussian_nll (see the reference in src/mlbook)')

def entropy(p: np.ndarray, eps: float=1e-12) -> np.ndarray:
    """Shannon entropy (nats) along the last axis. (..., K) -> (...)."""
    raise NotImplementedError('TODO: implement entropy (see the reference in src/mlbook)')

def predictive_entropy(P: np.ndarray) -> np.ndarray:
    """P (M, N, K) stacked member probabilities -> H[mean_m p_m] of shape (N,)."""
    raise NotImplementedError('TODO: implement predictive_entropy (see the reference in src/mlbook)')

def expected_entropy(P: np.ndarray) -> np.ndarray:
    """P (M, N, K) -> (1/M) sum_m H[p_m], shape (N,) (aleatoric part)."""
    raise NotImplementedError('TODO: implement expected_entropy (see the reference in src/mlbook)')

def mutual_information(P: np.ndarray) -> np.ndarray:
    """BALD score: H[p_bar] - E_m H[p_m]. P (M, N, K) -> (N,)."""
    raise NotImplementedError('TODO: implement mutual_information (see the reference in src/mlbook)')

def enable_mc_dropout(model: nn.Module) -> None:
    """Put only Dropout layers in train mode (BatchNorm etc. stay in eval)."""
    raise NotImplementedError('TODO: implement enable_mc_dropout (see the reference in src/mlbook)')

@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int=20) -> np.ndarray:
    """Stochastic forward passes with dropout on. x (N, d) -> probs (M, N, K)."""
    raise NotImplementedError('TODO: implement mc_dropout_predict (see the reference in src/mlbook)')

@torch.no_grad()
def ensemble_predict(models: list[nn.Module], x: torch.Tensor) -> np.ndarray:
    """Deep ensemble: one pass per member. x (N, d) -> probs (M, N, K)."""
    raise NotImplementedError('TODO: implement ensemble_predict (see the reference in src/mlbook)')
