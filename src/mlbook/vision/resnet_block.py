"""Residual, pre-activation, bottleneck, squeeze-excitation and MBConv blocks (PyTorch).

Every block maps ``(B, C_in, H, W) → (B, C_out, H/s, W/s)`` and, when shapes match,
computes ``y = x + F(x)`` so that ``∂y/∂x = I + ∂F/∂x`` — the identity term is the
gradient highway that makes 100+ layer training work.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """ResNet-18/34 block: ``y = ReLU(x + BN(conv3x3(ReLU(BN(conv3x3(x))))))``.

    When ``stride != 1`` or channels change, the shortcut is a 1×1 stride-``s`` conv + BN
    (ResNet "option B") so the two branches have equal shape.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        # zero-init the last BN's γ so the block starts as the identity (Goyal et al. 2017)
        nn.init.zeros_(self.bn2.weight)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.relu(self.bn1(self.conv1(x)))  # (B, C_out, H/s, W/s)
        residual = self.bn2(self.conv2(residual))  # (B, C_out, H/s, W/s)  F(x)
        identity = self.shortcut(x)  # (B, C_out, H/s, W/s)
        return torch.relu(identity + residual)  # (B, C_out, H/s, W/s)  y = x + F(x)


class PreActBlock(nn.Module):
    """Pre-activation block (He et al. 2016b): ``y = x + conv(ReLU(BN(conv(ReLU(BN(x))))))``.

    No ReLU after the addition, so the identity path is *clean* all the way from
    input to output: ``x_L = x_l + Σ_{i=l}^{L-1} F(x_i)`` holds exactly.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(self.bn1(x))  # (B, C_in, H, W)
        residual = self.conv1(pre)  # (B, C_out, H/s, W/s)
        residual = self.conv2(torch.relu(self.bn2(residual)))  # (B, C_out, H/s, W/s)
        return self.shortcut(x) + residual  # (B, C_out, H/s, W/s)  no post-add nonlinearity


class Bottleneck(nn.Module):
    """ResNet-50 block: 1×1 reduce → 3×3 → 1×1 expand (×4), residual around all three.

    The 1×1 convs make the expensive 3×3 run at ``C_out/4`` width:
    MACs ≈ H W (C·C/4 + 9·(C/4)² + C/4·C) vs H W · 9 C² for two plain 3×3 convs.
    """

    expansion = 4

    def __init__(self, c_in: int, c_mid: int, stride: int = 1) -> None:
        super().__init__()
        c_out = c_mid * self.expansion
        self.conv1 = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, 3, stride=stride, padding=1, bias=False)  # stride on the 3×3 (v1.5)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        nn.init.zeros_(self.bn3.weight)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))  # (B, C_mid, H, W)
        h = torch.relu(self.bn2(self.conv2(h)))  # (B, C_mid, H/s, W/s)
        h = self.bn3(self.conv3(h))  # (B, 4·C_mid, H/s, W/s)
        return torch.relu(self.shortcut(x) + h)  # (B, 4·C_mid, H/s, W/s)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al. 2018): per-channel gates from global context.

    ``s = σ(W₂ ReLU(W₁ · GAP(x)))``, ``y = s ⊙ x``.  Cost is tiny (two small FCs);
    gain is channel attention conditioned on the whole image.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))  # (B, C)  squeeze: global average pool
        gate = torch.sigmoid(self.fc2(torch.relu(self.fc1(pooled))))  # (B, C)  excitation
        return x * gate[:, :, None, None]  # (B, C, H, W)  rescale each channel


class MBConv(nn.Module):
    """Inverted residual with SE (MobileNetV2/V3, EfficientNet): expand → depthwise → SE → project.

    (B, C_in, H, W) → 1×1 expand to ``t·C_in`` → k×k depthwise (stride s) → SE → 1×1 project
    to C_out (linear, no activation).  Residual only when ``s == 1`` and ``C_in == C_out``.
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, stride: int = 1, expand: int = 4, se_reduction: int = 4) -> None:
        super().__init__()
        c_exp = c_in * expand
        self.expand = nn.Sequential(nn.Conv2d(c_in, c_exp, 1, bias=False), nn.BatchNorm2d(c_exp), nn.SiLU())
        self.depthwise = nn.Sequential(
            nn.Conv2d(c_exp, c_exp, k, stride=stride, padding=k // 2, groups=c_exp, bias=False),
            nn.BatchNorm2d(c_exp),
            nn.SiLU(),
        )
        self.se = SEBlock(c_exp, reduction=se_reduction * expand)
        self.project = nn.Sequential(nn.Conv2d(c_exp, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.use_residual = stride == 1 and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)  # (B, t·C_in, H, W)
        h = self.depthwise(h)  # (B, t·C_in, H/s, W/s)
        h = self.se(h)  # (B, t·C_in, H/s, W/s)
        h = self.project(h)  # (B, C_out, H/s, W/s)  linear bottleneck
        return x + h if self.use_residual else h  # (B, C_out, H/s, W/s)


class TinyResNet(nn.Module):
    """Stem + 3 stages of ``BasicBlock`` + GAP head; (B, C, H, W) → (B, K)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, width: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.stage1 = BasicBlock(width, width, stride=1)
        self.stage2 = BasicBlock(width, 2 * width, stride=2)
        self.stage3 = BasicBlock(2 * width, 4 * width, stride=2)
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)  # (B, w, H, W)
        f = self.stage3(self.stage2(self.stage1(f)))  # (B, 4w, H/4, W/4)
        return self.head(f.mean(dim=(2, 3)))  # (B, K)
