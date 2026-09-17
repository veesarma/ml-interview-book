"""Masked autoencoder (MAE) for images: random masking, an encoder on the *visible* patches only,
a light decoder that reconstructs every patch, and a loss on the masked patches only.

Shapes: image (B, C, H, W) → patches (B, N, P²·C) with N = (H/P)(W/P).
The Transformer blocks are ``nn.TransformerEncoderLayer`` (the attention maths is derived in
Part V, chapter 3); this module's content is the masking and the asymmetric design.
"""

from __future__ import annotations

import torch
from torch import nn


def patchify(img: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) → (B, N, P·P·C) with N = (H/P)·(W/P), row-major over the patch grid."""
    B, C, H, W = img.shape
    h, w = H // patch, W // patch
    x = img.reshape(B, C, h, patch, w, patch)                     # (B, C, h, P, w, P)
    x = x.permute(0, 2, 4, 3, 5, 1)                               # (B, h, w, P, P, C)
    x = x.reshape(B, h * w, patch * patch * C)                    # (B, N, P·P·C)
    return x


def unpatchify(x: torch.Tensor, patch: int, C: int, H: int, W: int) -> torch.Tensor:
    """Inverse of ``patchify``: (B, N, P·P·C) → (B, C, H, W)."""
    B = x.shape[0]
    h, w = H // patch, W // patch
    x = x.reshape(B, h, w, patch, patch, C)                       # (B, h, w, P, P, C)
    x = x.permute(0, 5, 1, 3, 2, 4)                               # (B, C, h, P, w, P)
    return x.reshape(B, C, H, W)                                  # (B, C, H, W)


def random_masking(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep a random subset of ``N_keep = round(N (1 − mask_ratio))`` tokens per example.

    Args:
        x: (B, N, D) patch tokens.
    Returns:
        x_visible: (B, N_keep, D);  mask: (B, N) with 1 = masked, 0 = kept;
        ids_restore: (B, N) permutation that puts [visible; masked] back into original order.
    """
    B, N, D = x.shape
    n_keep = int(round(N * (1.0 - mask_ratio)))
    noise = torch.rand(B, N)                                      # (B, N) one random key per token
    ids_shuffle = noise.argsort(dim=1)                            # (B, N) ascending noise = random perm
    ids_restore = ids_shuffle.argsort(dim=1)                      # (B, N) inverse permutation
    ids_keep = ids_shuffle[:, :n_keep]                            # (B, N_keep)
    x_visible = torch.gather(x, 1, ids_keep[:, :, None].expand(-1, -1, D))  # (B, N_keep, D)
    mask = torch.ones(B, N)                                       # (B, N)
    mask[:, :n_keep] = 0.0                                        # first n_keep of the *shuffled* order are kept
    mask = torch.gather(mask, 1, ids_restore)                     # (B, N) back in original token order
    return x_visible, mask, ids_restore


class MAE(nn.Module):
    """Tiny MAE.  forward(img) → (loss, pred (B, N, P·P·C), mask (B, N))."""

    def __init__(self, img_size: int = 16, patch: int = 4, in_chans: int = 1, d_enc: int = 64,
                 d_dec: int = 32, depth_enc: int = 2, depth_dec: int = 1, n_heads: int = 4,
                 mask_ratio: float = 0.75) -> None:
        super().__init__()
        self.patch, self.C, self.H, self.W = patch, in_chans, img_size, img_size
        self.n_patches = (img_size // patch) ** 2
        d_patch = patch * patch * in_chans
        self.mask_ratio = mask_ratio
        # encoder (operates on visible tokens only)
        self.patch_embed = nn.Linear(d_patch, d_enc)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.n_patches, d_enc))        # (1, N, d_enc)
        enc_layer = nn.TransformerEncoderLayer(d_enc, n_heads, 2 * d_enc, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth_enc)
        # decoder (light; sees mask tokens + encoded visible tokens)
        self.enc_to_dec = nn.Linear(d_enc, d_dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))                  # (1, 1, d_dec)
        self.pos_dec = nn.Parameter(torch.zeros(1, self.n_patches, d_dec))        # (1, N, d_dec)
        dec_layer = nn.TransformerEncoderLayer(d_dec, n_heads, 2 * d_dec, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, depth_dec)
        self.head = nn.Linear(d_dec, d_patch)
        nn.init.normal_(self.pos_enc, std=0.02)
        nn.init.normal_(self.pos_dec, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(patchify(img, self.patch))      # (B, N, d_enc)
        tokens = tokens + self.pos_enc                            # (B, N, d_enc)  position *before* masking
        x_vis, mask, ids_restore = random_masking(tokens, self.mask_ratio)  # (B, N_keep, d_enc), (B, N), (B, N)
        latent = self.encoder(x_vis)                              # (B, N_keep, d_enc)  ~25% of the tokens
        return latent, mask, ids_restore

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        B, n_keep, _ = latent.shape
        x = self.enc_to_dec(latent)                               # (B, N_keep, d_dec)
        n_masked = self.n_patches - n_keep
        mask_tokens = self.mask_token.expand(B, n_masked, -1)     # (B, N_masked, d_dec)
        x = torch.cat([x, mask_tokens], dim=1)                    # (B, N, d_dec) in shuffled order
        x = torch.gather(x, 1, ids_restore[:, :, None].expand(-1, -1, x.shape[2]))  # (B, N, d_dec) original order
        x = x + self.pos_dec                                      # (B, N, d_dec)
        x = self.decoder(x)                                       # (B, N, d_dec)
        return self.head(x)                                       # (B, N, P·P·C)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.encode(img)
        pred = self.decode(latent, ids_restore)                   # (B, N, P·P·C)
        target = patchify(img, self.patch)                        # (B, N, P·P·C)
        loss = masked_reconstruction_loss(pred, target, mask)
        return loss, pred, mask


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE averaged over *masked* patches only:  Σ_i mask_i · mean_j (pred_ij − target_ij)² / Σ_i mask_i.

    Args:
        pred, target: (B, N, D).  mask: (B, N) with 1 = masked.
    """
    per_patch = ((pred - target) ** 2).mean(dim=2)                # (B, N)
    return (per_patch * mask).sum() / mask.sum().clamp(min=1.0)
