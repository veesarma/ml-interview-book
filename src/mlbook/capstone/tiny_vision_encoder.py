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
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} is not divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.n_tokens = self.grid * self.grid
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, d_v)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_v))  # (1, N_v, d_v)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape                                        # (B, C, H, W)
        P, G = self.patch_size, self.grid
        x = images.reshape(B, C, G, P, G, P)                             # (B, C, G, P, G, P)
        x = x.permute(0, 2, 4, 1, 3, 5)                                  # (B, G, G, C, P, P)
        x = x.reshape(B, G * G, C * P * P)                               # (B, N_v, C*P*P)
        tokens = self.proj(x)                                            # (B, N_v, d_v)
        return tokens + self.pos_embed                                   # (B, N_v, d_v)


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with three separate Q/K/V projections.

    Input and output ``(B, N, d)``. Implements
    ``softmax(Q K^T / sqrt(d_head)) V`` per head, then a linear merge.
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"width {d} is not divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape                                                # (B, N, d)
        q = self.q_proj(x)                                               # (B, N, d)
        k = self.k_proj(x)                                               # (B, N, d)
        v = self.v_proj(x)                                               # (B, N, d)
        q = q.reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)   # (B, H_v, N, d_head)
        k = k.reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)   # (B, H_v, N, d_head)
        v = v.reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)   # (B, H_v, N, d_head)
        scores = q @ k.transpose(-2, -1) / self.d_head**0.5              # (B, H_v, N, N)
        weights = torch.softmax(scores, dim=-1)                          # (B, H_v, N, N)
        context = weights @ v                                            # (B, H_v, N, d_head)
        context = context.transpose(1, 2).reshape(B, N, d)               # (B, N, d)
        return self.out_proj(context)                                    # (B, N, d)


class VisionBlock(nn.Module):
    """Pre-norm Transformer block: ``x + attn(ln(x))`` then ``x + mlp(ln(x))``. ``(B, N, d) -> (B, N, d)``."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = SelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d),
            nn.GELU(),
            nn.Linear(mlp_ratio * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))                                   # (B, N, d)
        x = x + self.mlp(self.ln2(x))                                    # (B, N, d)
        return x                                                         # (B, N, d)


class TinyVisionEncoder(nn.Module):
    """Patch embed + ``n_layers`` attention blocks + a final LayerNorm.

    Input  ``(B, C, H, W)``.
    Output ``(B, N_v, d_v)`` -- one token per image patch, no ``[CLS]`` token,
    because the projector in :mod:`mlbook.capstone.projector` consumes all of them.
    """

    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 3,
        d_v: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_v)
        self.blocks = nn.ModuleList([VisionBlock(d_v, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_v)
        self.n_tokens = self.patch_embed.n_tokens
        self.d_v = d_v

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)                                     # (B, N_v, d_v)
        for block in self.blocks:
            x = block(x)                                                 # (B, N_v, d_v)
        return self.ln_f(x)                                              # (B, N_v, d_v)
