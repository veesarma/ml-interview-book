# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/clip.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k clip -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/clip --force

"""Mini-CLIP: two small encoders, projection heads, the symmetric InfoNCE loss,
SigLIP's pairwise sigmoid loss, and zero-shot classification.

    v_i = normalise(g_v(f_vision(x_i)))   # (B, d_e)
    t_j = normalise(g_t(f_text(c_j)))     # (B, d_e)
    S   = V T^T / τ                        # (B, B), S_ij = v_i · t_j / τ
    L_clip = 1/2 [ CE(S, arange(B)) + CE(S^T, arange(B)) ]
    L_siglip = -1/B Σ_ij log σ( z_ij (t' v_i·t_j + b) ),  z_ij = +1 on the diagonal, -1 elsewhere
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .attention_block import TransformerBlock
from .patch_embed import PatchEmbedLinear

class TinyImageEncoder(nn.Module):
    """Patch-embed + Transformer blocks + mean pool: (B, C, H, W) -> (B, d)."""

    def __init__(self, image_size: int, patch: int, in_channels: int, d: int, depth: int, n_heads: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyTextEncoder(nn.Module):
    """Token embedding + Transformer blocks + mean pool over non-pad tokens: (B, T) -> (B, d)."""

    def __init__(self, vocab: int, max_len: int, d: int, depth: int, n_heads: int, pad_id: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class MiniCLIP(nn.Module):
    """Dual encoder with projection heads and a learnable log-temperature.

    ``encode_image``: (B, C, H, W) -> (B, d_e) unit vectors; ``encode_text``: (B, T) -> (B, d_e).
    ``forward`` returns logits_per_image (B, B) = V T^T * exp(logit_scale).
    """

    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module, d_img: int, d_txt: int, d_embed: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement encode_image (see the reference in src/mlbook)')

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement encode_text (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def clip_loss(logits_per_image: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE. ``logits_per_image`` (B, B) with matched pairs on the diagonal. Returns scalar."""
    raise NotImplementedError('TODO: implement clip_loss (see the reference in src/mlbook)')

def siglip_loss(v: torch.Tensor, t: torch.Tensor, log_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """SigLIP pairwise sigmoid loss. ``v``, ``t`` (B, d_e) unit vectors; ``log_t``, ``bias`` scalars. Scalar.

    Every (i, j) pair is an independent binary problem: positive iff i == j.
    No softmax over the batch, so each pair's gradient does not depend on batch size.
    """
    raise NotImplementedError('TODO: implement siglip_loss (see the reference in src/mlbook)')

@torch.no_grad()
def zero_shot_classify(model: MiniCLIP, images: torch.Tensor, class_prompt_ids: torch.Tensor) -> torch.Tensor:
    """Zero-shot logits (B, K): cosine similarity between image embeddings and K class-prompt embeddings.

    ``class_prompt_ids`` (K, T) — one tokenised prompt per class (average several templates
    per class before normalising for the CLIP "prompt ensembling" trick).
    """
    raise NotImplementedError('TODO: implement zero_shot_classify (see the reference in src/mlbook)')
