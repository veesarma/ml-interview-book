# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/interp/integrated_gradients.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k integrated_gradients -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py interp/integrated_gradients --force

"""Integrated gradients (Sundararajan, Taly & Yan 2017).

    IG_i(x) = (x_i - x'_i) * integral_0^1  d f(x' + t (x - x')) / d x_i  dt

Completeness (fundamental theorem of calculus along the straight path):

    sum_i IG_i(x) = f(x) - f(x')

Riemann approximation with m steps (midpoint rule): t_k = (k - 0.5)/m.
"""
from __future__ import annotations
import torch
from torch import nn

def integrated_gradients(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor, target: int, steps: int=64) -> torch.Tensor:
    """IG attribution for one input x (any shape); baseline has the same shape."""
    raise NotImplementedError('TODO: implement integrated_gradients (see the reference in src/mlbook)')

def completeness_gap(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor, target: int, steps: int=64) -> float:
    """| sum_i IG_i - (f(x) - f(x')) |, should shrink as steps grows."""
    raise NotImplementedError('TODO: implement completeness_gap (see the reference in src/mlbook)')
