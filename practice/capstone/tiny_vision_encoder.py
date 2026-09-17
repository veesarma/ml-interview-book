# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/tiny_vision_encoder.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tiny_vision_encoder -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/tiny_vision_encoder --force

"""Canon #60, part 1 -- the vision tower: patch embedding plus explicit attention blocks.

This is canon #39 (tiny ViT) rebuilt inside the capstone so the capstone has no
dependency on the rest of the package: an interviewer who asks you to "sketch a
VLM" wants to see the tower written out, not imported.

Shape vocabulary used throughout:

* ``B``      batch
* ``C,H,W``  image channels, height, width
* ``P``      patch side in pixels
* ``N_v``    number of visual tokens, ``(H/P) * (W/P)``
* ``d_v``    vision width
* ``H_v``    vision heads, ``d_head = d_v / H_v``
"""
from __future__ import annotations
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Cut an image into non-overlapping patches and linearly embed each one.

    Input  ``(B, C, H, W)``.
    Output ``(B, N_v, d_v)`` with ``N_v = (H // P) * (W // P)``.

    Implements ``z_i = flatten(patch_i) W + b`` -- the same thing as a
    ``Conv2d(C, d_v, kernel_size=P, stride=P)``, written with ``unfold`` so the
    reshape is visible.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_v: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with three separate Q/K/V projections.

    Input and output ``(B, N, d)``. Implements
    ``softmax(Q K^T / sqrt(d_head)) V`` per head, then a linear merge.
    """

    def __init__(self, d: int, n_heads: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class VisionBlock(nn.Module):
    """Pre-norm Transformer block: ``x + attn(ln(x))`` then ``x + mlp(ln(x))``. ``(B, N, d) -> (B, N, d)``."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int=2) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyVisionEncoder(nn.Module):
    """Patch embed + ``n_layers`` attention blocks + a final LayerNorm.

    Input  ``(B, C, H, W)``.
    Output ``(B, N_v, d_v)`` -- one token per image patch, no ``[CLS]`` token,
    because the projector in :mod:`mlbook.capstone.projector` consumes all of them.
    """

    def __init__(self, image_size: int=16, patch_size: int=4, in_channels: int=3, d_v: int=32, n_heads: int=2, n_layers: int=2) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
