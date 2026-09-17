# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/vit.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k vit -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/vit --force

"""A tiny Vision Transformer from scratch.

    tokens = PatchEmbed(image)            # (B, N, d),  N = HW / P^2
    x      = [CLS; tokens] + pos          # (B, N+1, d)
    x      = Block_L(... Block_1(x))       # (B, N+1, d)
    logits = Head(LN(x)[:, 0])            # (B, K)     (or mean over tokens with pool="gap")

Optionally adds *register* tokens (extra learned tokens that carry global
information and are discarded at the output) and a DeiT-style distillation token.
"""
from __future__ import annotations
import torch
from torch import nn
from .attention_block import TransformerBlock
from .patch_embed import PatchEmbedLinear

class TinyViT(nn.Module):
    """Vision Transformer classifier: (B, C, H, W) -> (B, K).

    ``pool="cls"`` reads the CLS token; ``pool="gap"`` averages the patch tokens.
    ``n_registers`` extra tokens are prepended after CLS and dropped at the end.
    """

    def __init__(self, image_size: int, patch: int, in_channels: int, d: int, depth: int, n_heads: int, n_classes: int, pool: str='cls', n_registers: int=0, mlp_ratio: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder: (B, C, H, W) -> (B, 1+R+N, d) normalised tokens."""
        raise NotImplementedError('TODO: implement tokens (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class DistillableViT(TinyViT):
    """DeiT-style ViT with a distillation token: returns (cls_logits, dist_logits), each (B, K).

    Train with ``CE(cls_logits, y) + CE(dist_logits, teacher_argmax)`` (hard distillation);
    at test time average the two logit vectors.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
