# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/vision/resnet_block.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k resnet_block -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py vision/resnet_block --force

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

    def __init__(self, c_in: int, c_out: int, stride: int=1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class PreActBlock(nn.Module):
    """Pre-activation block (He et al. 2016b): ``y = x + conv(ReLU(BN(conv(ReLU(BN(x))))))``.

    No ReLU after the addition, so the identity path is *clean* all the way from
    input to output: ``x_L = x_l + Σ_{i=l}^{L-1} F(x_i)`` holds exactly.
    """

    def __init__(self, c_in: int, c_out: int, stride: int=1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class Bottleneck(nn.Module):
    """ResNet-50 block: 1×1 reduce → 3×3 → 1×1 expand (×4), residual around all three.

    The 1×1 convs make the expensive 3×3 run at ``C_out/4`` width:
    MACs ≈ H W (C·C/4 + 9·(C/4)² + C/4·C) vs H W · 9 C² for two plain 3×3 convs.
    """
    expansion = 4

    def __init__(self, c_in: int, c_mid: int, stride: int=1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al. 2018): per-channel gates from global context.

    ``s = σ(W₂ ReLU(W₁ · GAP(x)))``, ``y = s ⊙ x``.  Cost is tiny (two small FCs);
    gain is channel attention conditioned on the whole image.
    """

    def __init__(self, channels: int, reduction: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class MBConv(nn.Module):
    """Inverted residual with SE (MobileNetV2/V3, EfficientNet): expand → depthwise → SE → project.

    (B, C_in, H, W) → 1×1 expand to ``t·C_in`` → k×k depthwise (stride s) → SE → 1×1 project
    to C_out (linear, no activation).  Residual only when ``s == 1`` and ``C_in == C_out``.
    """

    def __init__(self, c_in: int, c_out: int, k: int=3, stride: int=1, expand: int=4, se_reduction: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyResNet(nn.Module):
    """Stem + 3 stages of ``BasicBlock`` + GAP head; (B, C, H, W) → (B, K)."""

    def __init__(self, in_channels: int=1, num_classes: int=4, width: int=16) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
