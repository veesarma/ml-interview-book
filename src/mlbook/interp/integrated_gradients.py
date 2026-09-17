"""Integrated gradients (Sundararajan, Taly & Yan 2017).

    IG_i(x) = (x_i - x'_i) * integral_0^1  d f(x' + t (x - x')) / d x_i  dt

Completeness (fundamental theorem of calculus along the straight path):

    sum_i IG_i(x) = f(x) - f(x')

Riemann approximation with m steps (midpoint rule): t_k = (k - 0.5)/m.
"""

from __future__ import annotations

import torch
from torch import nn


def integrated_gradients(
    model: nn.Module, x: torch.Tensor, baseline: torch.Tensor, target: int, steps: int = 64
) -> torch.Tensor:
    """IG attribution for one input x (any shape); baseline has the same shape."""
    alphas = (torch.arange(steps, dtype=x.dtype) + 0.5) / steps  # (m,)
    delta = x - baseline  # same shape as x
    path = baseline.unsqueeze(0) + alphas.view(-1, *([1] * x.dim())) * delta.unsqueeze(0)  # (m, *x.shape)
    path = path.detach().requires_grad_(True)
    scores = model(path)[:, target].sum()  # scalar; sum over the m path points
    (grads,) = torch.autograd.grad(scores, path)  # (m, *x.shape)
    return delta * grads.mean(dim=0)  # same shape as x


def completeness_gap(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor, target: int, steps: int = 64) -> float:
    """| sum_i IG_i - (f(x) - f(x')) |, should shrink as steps grows."""
    ig = integrated_gradients(model, x, baseline, target, steps)
    with torch.no_grad():
        fx = model(x.unsqueeze(0))[0, target]
        fb = model(baseline.unsqueeze(0))[0, target]
    return float((ig.sum() - (fx - fb)).abs())
