"""Gradient accumulation: k micro-batches of size b == one batch of size k b.

For a mean-reduced loss over a batch of size B = k b, split into micro-batches M_1..M_k:

    L = (1/B) sum_{i in batch} l_i = (1/k) sum_j [ (1/b) sum_{i in M_j} l_i ] = (1/k) sum_j L_j

so accumulating grad(L_j) / k over the k micro-batches gives grad(L) exactly. The
only caveat is any layer whose forward depends on the *batch composition*:
BatchNorm computes statistics over the micro-batch, so its gradient (and forward)
differ from the full-batch version. LayerNorm/RMSNorm are per-example and are fine.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def full_batch_grads(model: nn.Module, loss_fn: Callable, X: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """Gradient of the mean loss over the full batch X: (B, ...)."""
    model.zero_grad()
    loss_fn(model(X), y).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def accumulated_grads(
    model: nn.Module, loss_fn: Callable, X: torch.Tensor, y: torch.Tensor, k: int
) -> list[torch.Tensor]:
    """Gradient accumulated over k equal micro-batches with loss scaled by 1/k."""
    model.zero_grad()
    for X_j, y_j in zip(torch.chunk(X, k), torch.chunk(y, k)):  # X_j: (B/k, ...)
        (loss_fn(model(X_j), y_j) / k).backward()  # grads accumulate in p.grad
    return [p.grad.detach().clone() for p in model.parameters()]


def max_abs_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def make_mlp(norm: str = "layernorm", d: int = 8, hidden: int = 16, seed: int = 0) -> nn.Module:
    """Small MLP whose middle normalisation is LayerNorm (accumulation-safe) or
    BatchNorm (accumulation-unsafe)."""
    torch.manual_seed(seed)
    norm_layer = nn.LayerNorm(hidden) if norm == "layernorm" else nn.BatchNorm1d(hidden)
    return nn.Sequential(nn.Linear(d, hidden), norm_layer, nn.Tanh(), nn.Linear(hidden, 1))
