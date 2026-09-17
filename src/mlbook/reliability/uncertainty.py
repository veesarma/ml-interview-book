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
    var = log_var.exp()  # (N,)
    return (0.5 * log_var + (y - mu) ** 2 / (2.0 * var)).mean()


def entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy (nats) along the last axis. (..., K) -> (...)."""
    return -np.sum(p * np.log(p + eps), axis=-1)


def predictive_entropy(P: np.ndarray) -> np.ndarray:
    """P (M, N, K) stacked member probabilities -> H[mean_m p_m] of shape (N,)."""
    return entropy(P.mean(axis=0))  # (N,)


def expected_entropy(P: np.ndarray) -> np.ndarray:
    """P (M, N, K) -> (1/M) sum_m H[p_m], shape (N,) (aleatoric part)."""
    return entropy(P).mean(axis=0)  # (N,)


def mutual_information(P: np.ndarray) -> np.ndarray:
    """BALD score: H[p_bar] - E_m H[p_m]. P (M, N, K) -> (N,)."""
    return np.clip(predictive_entropy(P) - expected_entropy(P), 0.0, None)  # (N,)


def enable_mc_dropout(model: nn.Module) -> None:
    """Put only Dropout layers in train mode (BatchNorm etc. stay in eval)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 20) -> np.ndarray:
    """Stochastic forward passes with dropout on. x (N, d) -> probs (M, N, K)."""
    enable_mc_dropout(model)
    outs = [torch.softmax(model(x), dim=-1) for _ in range(n_samples)]  # M x (N, K)
    model.eval()
    return torch.stack(outs, dim=0).numpy()  # (M, N, K)


@torch.no_grad()
def ensemble_predict(models: list[nn.Module], x: torch.Tensor) -> np.ndarray:
    """Deep ensemble: one pass per member. x (N, d) -> probs (M, N, K)."""
    for m in models:
        m.eval()
    outs = [torch.softmax(m(x), dim=-1) for m in models]  # M x (N, K)
    return torch.stack(outs, dim=0).numpy()  # (M, N, K)
