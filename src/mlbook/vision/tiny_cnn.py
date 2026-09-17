"""A tiny CNN classifier in PyTorch: conv → BN → ReLU blocks, pooling, a linear head.

Input ``(B, C, H, W)``; output logits ``(B, K)``.  Small enough to train on CPU in a
test, structured like the stem+stages+head pattern every ImageNet CNN shares.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    """``y = ReLU(BN(Conv_{k×k}(x)))`` — the unit every CNN since VGG/BN is built from.

    Shapes: (B, C_in, H, W) → (B, C_out, H/s, W/s) with "same" padding ``k//2``.
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)  # bias lives in BN's β, so the conv has none
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)  # (B, C_out, H/s, W/s)
        y = self.bn(y)  # (B, C_out, H/s, W/s)
        return self.act(y)  # (B, C_out, H/s, W/s)


class TinyCNN(nn.Module):
    """Stem + two stages + global-average-pool head.

    (B, C, H, W) → stem (B, 16, H, W) → stage1 (B, 32, H/2, W/2) → stage2 (B, 64, H/4, W/4)
    → GAP (B, 64) → logits (B, K).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, width: int = 16) -> None:
        super().__init__()
        self.stem = ConvBNReLU(in_channels, width, k=3, stride=1)
        self.stage1 = nn.Sequential(ConvBNReLU(width, 2 * width, stride=2), ConvBNReLU(2 * width, 2 * width))
        self.stage2 = nn.Sequential(ConvBNReLU(2 * width, 4 * width, stride=2), ConvBNReLU(4 * width, 4 * width))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)  # (B, w, H, W)
        f = self.stage1(f)  # (B, 2w, H/2, W/2)
        f = self.stage2(f)  # (B, 4w, H/4, W/4)
        pooled = self.pool(f).flatten(1)  # (B, 4w)  global average pool
        return self.head(pooled)  # (B, K) logits


def synthetic_shapes(n: int, size: int = 32, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A 4-class toy dataset: filled square / hollow square / horizontal bar / vertical bar.

    Returns:
        images (n, 1, size, size) in [0, 1] with noise; labels (n,) in {0,1,2,3}.
    """
    g = generator if generator is not None else torch.Generator().manual_seed(0)
    images = 0.1 * torch.rand(n, 1, size, size, generator=g)  # (n, 1, S, S) noise floor
    labels = torch.randint(0, 4, (n,), generator=g)  # (n,)
    for idx in range(n):
        top = int(torch.randint(4, size // 2, (1,), generator=g))  # random position
        left = int(torch.randint(4, size // 2, (1,), generator=g))
        side = int(torch.randint(8, size // 2, (1,), generator=g))
        y = labels[idx].item()
        if y == 0:  # filled square
            images[idx, 0, top : top + side, left : left + side] = 1.0
        elif y == 1:  # hollow square (2-px border)
            images[idx, 0, top : top + side, left : left + side] = 1.0
            images[idx, 0, top + 2 : top + side - 2, left + 2 : left + side - 2] = 0.1
        elif y == 2:  # horizontal bar
            images[idx, 0, top : top + 3, left : left + side] = 1.0
        else:  # vertical bar
            images[idx, 0, top : top + side, left : left + 3] = 1.0
    return images, labels
