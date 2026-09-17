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
    x = x.detach().clone().requires_grad_(True)  # (C, H, W)
    score = model(x.unsqueeze(0))[0, target]  # scalar: class logit
    (grad,) = torch.autograd.grad(score, x)  # (C, H, W)
    return grad.detach()


def smoothgrad(model: nn.Module, x: torch.Tensor, target: int, n_samples: int = 32, sigma: float = 0.15) -> torch.Tensor:
    """Average vanilla gradient over n Gaussian-perturbed copies of x. Same shape as x."""
    acc = torch.zeros_like(x)  # (C, H, W)
    for _ in range(n_samples):
        noisy = x + sigma * torch.randn_like(x)  # (C, H, W)
        acc += vanilla_gradient(model, noisy, target)
    return acc / n_samples


def saliency_map(grad: torch.Tensor) -> torch.Tensor:
    """Collapse channels: max_c |grad| -> (H, W), normalised to [0, 1]."""
    m = grad.abs().amax(dim=0)  # (H, W)
    return m / (m.max() + 1e-12)
