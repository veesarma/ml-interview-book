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

    def __init__(self, n_classes: int = 2, width: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.fc = nn.Linear(width, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))  # (B, K, H, W)
        return F.relu(self.conv2(h))  # (B, K, H, W)  <- the map Grad-CAM uses

    def head(self, A: torch.Tensor) -> torch.Tensor:
        return self.fc(A.mean(dim=(2, 3)))  # (B, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # (B, n_classes)


def grad_cam(model: TinyCNN, x: torch.Tensor, target: int) -> torch.Tensor:
    """Grad-CAM heatmap for one image x: (1, H, W) -> (H, W) in [0, 1]."""
    A = model.features(x.unsqueeze(0))  # (1, K, H', W')
    A.retain_grad()
    score = model.head(A)[0, target]  # scalar
    (dA,) = torch.autograd.grad(score, A)  # (1, K, H', W')
    alpha = dA.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)  pooled gradients
    cam = F.relu((alpha * A).sum(dim=1, keepdim=True))  # (1, 1, H', W')
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)  # (1, 1, H, W)
    cam = cam[0, 0].detach()  # (H, W)
    return cam / (cam.max() + 1e-12)
