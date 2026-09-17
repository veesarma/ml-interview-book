# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/vision/tiny_cnn.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tiny_cnn -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py vision/tiny_cnn --force

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

    def __init__(self, c_in: int, c_out: int, k: int=3, stride: int=1, groups: int=1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyCNN(nn.Module):
    """Stem + two stages + global-average-pool head.

    (B, C, H, W) → stem (B, 16, H, W) → stage1 (B, 32, H/2, W/2) → stage2 (B, 64, H/4, W/4)
    → GAP (B, 64) → logits (B, K).
    """

    def __init__(self, in_channels: int=1, num_classes: int=4, width: int=16) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def synthetic_shapes(n: int, size: int=32, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """A 4-class toy dataset: filled square / hollow square / horizontal bar / vertical bar.

    Returns:
        images (n, 1, size, size) in [0, 1] with noise; labels (n,) in {0,1,2,3}.
    """
    raise NotImplementedError('TODO: implement synthetic_shapes (see the reference in src/mlbook)')
