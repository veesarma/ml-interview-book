# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/nn/normalization_torch.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k normalization_torch -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py nn/normalization_torch --force

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

    def __init__(self, d: int, eps: float=1e-05, momentum: float=0.1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class LayerNorm(nn.Module):
    """``y = gamma * (x - mu) / sqrt(var + eps) + beta`` with stats over the last axis of ``(..., D)``."""

    def __init__(self, d: int, eps: float=1e-05) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class RMSNorm(nn.Module):
    """``y = g * x / sqrt(mean(x^2) + eps)`` over the last axis - the Llama / T5 norm.

    The RMS is computed in float32 even for bf16 inputs (as Llama does), then
    cast back, because ``mean(x^2)`` in bf16 loses several bits.
    """

    def __init__(self, d: int, eps: float=1e-06) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
