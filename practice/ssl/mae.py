# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/ssl/mae.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k mae -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py ssl/mae --force

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
    raise NotImplementedError('TODO: implement patchify (see the reference in src/mlbook)')

def unpatchify(x: torch.Tensor, patch: int, C: int, H: int, W: int) -> torch.Tensor:
    """Inverse of ``patchify``: (B, N, P·P·C) → (B, C, H, W)."""
    raise NotImplementedError('TODO: implement unpatchify (see the reference in src/mlbook)')

def random_masking(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep a random subset of ``N_keep = round(N (1 − mask_ratio))`` tokens per example.

    Args:
        x: (B, N, D) patch tokens.
    Returns:
        x_visible: (B, N_keep, D);  mask: (B, N) with 1 = masked, 0 = kept;
        ids_restore: (B, N) permutation that puts [visible; masked] back into original order.
    """
    raise NotImplementedError('TODO: implement random_masking (see the reference in src/mlbook)')

class MAE(nn.Module):
    """Tiny MAE.  forward(img) → (loss, pred (B, N, P·P·C), mask (B, N))."""

    def __init__(self, img_size: int=16, patch: int=4, in_chans: int=1, d_enc: int=64, d_dec: int=32, depth_enc: int=2, depth_dec: int=1, n_heads: int=4, mask_ratio: float=0.75) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def encode(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE averaged over *masked* patches only:  Σ_i mask_i · mean_j (pred_ij − target_ij)² / Σ_i mask_i.

    Args:
        pred, target: (B, N, D).  mask: (B, N) with 1 = masked.
    """
    raise NotImplementedError('TODO: implement masked_reconstruction_loss (see the reference in src/mlbook)')
