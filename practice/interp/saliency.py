# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/interp/saliency.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k saliency -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py interp/saliency --force

"""Gradient saliency: vanilla gradients (Simonyan et al. 2014) and SmoothGrad
(Smilkov et al. 2017).

    vanilla:    S(x) = d f_c(x) / d x                         (same shape as x)
    SmoothGrad: S(x) = (1/n) sum_i d f_c(x + eps_i) / d x,  eps_i ~ N(0, sigma^2 I)

where f_c is the pre-softmax score of class c.
"""
from __future__ import annotations
import torch
from torch import nn

def vanilla_gradient(model: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    """d f_target / d x for a single input x: (C, H, W) or (d,) -> same shape."""
    raise NotImplementedError('TODO: implement vanilla_gradient (see the reference in src/mlbook)')

def smoothgrad(model: nn.Module, x: torch.Tensor, target: int, n_samples: int=32, sigma: float=0.15) -> torch.Tensor:
    """Average vanilla gradient over n Gaussian-perturbed copies of x. Same shape as x."""
    raise NotImplementedError('TODO: implement smoothgrad (see the reference in src/mlbook)')

def saliency_map(grad: torch.Tensor) -> torch.Tensor:
    """Collapse channels: max_c |grad| -> (H, W), normalised to [0, 1]."""
    raise NotImplementedError('TODO: implement saliency_map (see the reference in src/mlbook)')
