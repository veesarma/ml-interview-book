"""``nn.Module`` versions of BatchNorm1d, LayerNorm and RMSNorm (Part III, chapter 5).

Written from the equations, not by subclassing the torch classes, so you can
see every buffer and every reduction. Autograd supplies the backward; the
NumPy module ``mlbook.nn.normalization`` shows it by hand.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class BatchNorm1d(nn.Module):
    """``y = gamma * (x - mu_B) / sqrt(var_B + eps) + beta`` over the batch axis of ``(N, D)``.

    Buffers ``running_mean``/``running_var`` (unbiased var, matching PyTorch)
    are updated in training and *used* in eval.
    """

    def __init__(self, d: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))  # (D,)
        self.beta = nn.Parameter(torch.zeros(d))  # (D,)
        self.register_buffer("running_mean", torch.zeros(d))  # (D,)
        self.register_buffer("running_var", torch.ones(d))  # (D,)
        self.eps, self.momentum = eps, momentum

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = x.mean(dim=0)  # (D,)
            var = x.var(dim=0, unbiased=False)  # (D,)
            with torch.no_grad():
                n = x.shape[0]
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var * n / max(n - 1, 1))
        else:
            mu, var = self.running_mean, self.running_var  # (D,), (D,)
        xhat = (x - mu) / torch.sqrt(var + self.eps)  # (N, D)
        return self.gamma * xhat + self.beta  # (N, D)


class LayerNorm(nn.Module):
    """``y = gamma * (x - mu) / sqrt(var + eps) + beta`` with stats over the last axis of ``(..., D)``."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))  # (D,)
        self.beta = nn.Parameter(torch.zeros(d))  # (D,)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(dim=-1, keepdim=True)  # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (..., 1)
        xhat = (x - mu) / torch.sqrt(var + self.eps)  # (..., D)
        return self.gamma * xhat + self.beta  # (..., D)


class RMSNorm(nn.Module):
    """``y = g * x / sqrt(mean(x^2) + eps)`` over the last axis - the Llama / T5 norm.

    The RMS is computed in float32 even for bf16 inputs (as Llama does), then
    cast back, because ``mean(x^2)`` in bf16 loses several bits.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))  # (D,)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()  # (..., D) in fp32
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        return self.g * (xf * inv_rms).to(x.dtype)  # (..., D)
