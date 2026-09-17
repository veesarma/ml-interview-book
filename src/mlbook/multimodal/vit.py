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

    def __init__(
        self,
        image_size: int,
        patch: int,
        in_channels: int,
        d: int,
        depth: int,
        n_heads: int,
        n_classes: int,
        pool: str = "cls",
        n_registers: int = 0,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if pool not in ("cls", "gap"):
            raise ValueError("pool must be 'cls' or 'gap'")
        self.pool = pool
        self.n_registers = n_registers
        self.grid = image_size // patch
        n_patches = self.grid * self.grid
        self.patch_embed = PatchEmbedLinear(in_channels, patch, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))  # (1, 1, d)
        self.registers = nn.Parameter(torch.zeros(1, n_registers, d))  # (1, R, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_registers + n_patches, d))  # (1, 1+R+N, d)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder: (B, C, H, W) -> (B, 1+R+N, d) normalised tokens."""
        B = x.shape[0]
        t = self.patch_embed(x)  # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        reg = self.registers.expand(B, -1, -1)  # (B, R, d)
        t = torch.cat([cls, reg, t], dim=1)  # (B, 1+R+N, d)
        t = t + self.pos_embed  # (B, 1+R+N, d)
        for block in self.blocks:
            t = block(t)  # (B, 1+R+N, d)
        return self.norm(t)  # (B, 1+R+N, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.tokens(x)  # (B, 1+R+N, d)
        if self.pool == "cls":
            feat = t[:, 0]  # (B, d)
        else:
            feat = t[:, 1 + self.n_registers :].mean(dim=1)  # (B, d)
        return self.head(feat)  # (B, K)


class DistillableViT(TinyViT):
    """DeiT-style ViT with a distillation token: returns (cls_logits, dist_logits), each (B, K).

    Train with ``CE(cls_logits, y) + CE(dist_logits, teacher_argmax)`` (hard distillation);
    at test time average the two logit vectors.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d = self.cls_token.shape[-1]
        self.dist_token = nn.Parameter(torch.zeros(1, 1, d))  # (1, 1, d)
        n_tot = self.pos_embed.shape[1] + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tot, d))  # (1, 2+R+N, d)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dist_head = nn.Linear(d, self.head.out_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        B = x.shape[0]
        t = self.patch_embed(x)  # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        dist = self.dist_token.expand(B, -1, -1)  # (B, 1, d)
        reg = self.registers.expand(B, -1, -1)  # (B, R, d)
        t = torch.cat([cls, dist, reg, t], dim=1) + self.pos_embed  # (B, 2+R+N, d)
        for block in self.blocks:
            t = block(t)  # (B, 2+R+N, d)
        t = self.norm(t)  # (B, 2+R+N, d)
        return self.head(t[:, 0]), self.dist_head(t[:, 1])  # (B, K), (B, K)
