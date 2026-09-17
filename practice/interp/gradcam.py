# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/interp/gradcam.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gradcam -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py interp/gradcam --force

"""Grad-CAM (Selvaraju et al. 2017) on a tiny CNN.

For the last conv feature map A in R^{K x H' x W'} and class score y_c:

    alpha_k = (1 / (H' W')) sum_{i,j} d y_c / d A_{k,i,j}        (global-average-pooled gradient)
    CAM     = ReLU( sum_k alpha_k A_k )                            (H', W')

then upsample to the input size. ReLU keeps only evidence *for* the class.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class TinyCNN(nn.Module):
    """conv -> relu -> conv -> relu -> global-avg-pool -> linear. Input (B, 1, H, W)."""

    def __init__(self, n_classes: int=2, width: int=8):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement features (see the reference in src/mlbook)')

    def head(self, A: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement head (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def grad_cam(model: TinyCNN, x: torch.Tensor, target: int) -> torch.Tensor:
    """Grad-CAM heatmap for one image x: (1, H, W) -> (H, W) in [0, 1]."""
    raise NotImplementedError('TODO: implement grad_cam (see the reference in src/mlbook)')
