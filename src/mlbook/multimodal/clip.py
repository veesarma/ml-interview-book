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
        super().__init__()
        n = (image_size // patch) ** 2
        self.embed = PatchEmbedLinear(in_channels, patch, d)
        self.pos = nn.Parameter(torch.zeros(1, n, d))  # (1, N, d)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.embed(x) + self.pos  # (B, N, d)
        for blk in self.blocks:
            t = blk(t)  # (B, N, d)
        return self.norm(t).mean(dim=1)  # (B, d)


class TinyTextEncoder(nn.Module):
    """Token embedding + Transformer blocks + mean pool over non-pad tokens: (B, T) -> (B, d)."""

    def __init__(self, vocab: int, max_len: int, d: int, depth: int, n_heads: int, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))  # (1, T_max, d)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        t = self.tok(ids) + self.pos[:, :T]  # (B, T, d)
        keep = (ids != self.pad_id).float()[:, None, None, :]  # (B, 1, 1, T): 1 keep, 0 pad key
        mask = (1.0 - keep) * -1e9  # (B, 1, 1, T) additive mask on keys
        for blk in self.blocks:
            t = blk(t, mask)  # (B, T, d)
        t = self.norm(t)  # (B, T, d)
        valid = (ids != self.pad_id).float()[:, :, None]  # (B, T, 1)
        return (t * valid).sum(1) / valid.sum(1).clamp(min=1.0)  # (B, d)


class MiniCLIP(nn.Module):
    """Dual encoder with projection heads and a learnable log-temperature.

    ``encode_image``: (B, C, H, W) -> (B, d_e) unit vectors; ``encode_text``: (B, T) -> (B, d_e).
    ``forward`` returns logits_per_image (B, B) = V T^T * exp(logit_scale).
    """

    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module, d_img: int, d_txt: int, d_embed: int) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.proj_img = nn.Linear(d_img, d_embed, bias=False)
        self.proj_txt = nn.Linear(d_txt, d_embed, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # scalar, τ = 0.07 at init
        self.max_logit_scale = math.log(100.0)  # CLIP clamps τ >= 0.01

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        v = self.proj_img(self.image_encoder(x))  # (B, d_e)
        return nn.functional.normalize(v, dim=-1)  # (B, d_e)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        t = self.proj_txt(self.text_encoder(ids))  # (B, d_e)
        return nn.functional.normalize(t, dim=-1)  # (B, d_e)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        v = self.encode_image(x)  # (B, d_e)
        t = self.encode_text(ids)  # (B, d_e)
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()  # scalar = 1/τ
        return scale * v @ t.T  # (B, B): row i = image i vs every text


def clip_loss(logits_per_image: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE. ``logits_per_image`` (B, B) with matched pairs on the diagonal. Returns scalar."""
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)  # (B,)
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)  # image -> text
    loss_t = nn.functional.cross_entropy(logits_per_image.T, labels)  # text -> image
    return 0.5 * (loss_i + loss_t)


def siglip_loss(v: torch.Tensor, t: torch.Tensor, log_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """SigLIP pairwise sigmoid loss. ``v``, ``t`` (B, d_e) unit vectors; ``log_t``, ``bias`` scalars. Scalar.

    Every (i, j) pair is an independent binary problem: positive iff i == j.
    No softmax over the batch, so each pair's gradient does not depend on batch size.
    """
    B = v.shape[0]
    logits = (v @ t.T) * log_t.exp() + bias  # (B, B)
    z = 2 * torch.eye(B, device=v.device) - 1  # (B, B): +1 diagonal, -1 off-diagonal
    return -nn.functional.logsigmoid(z * logits).sum() / B


@torch.no_grad()
def zero_shot_classify(model: MiniCLIP, images: torch.Tensor, class_prompt_ids: torch.Tensor) -> torch.Tensor:
    """Zero-shot logits (B, K): cosine similarity between image embeddings and K class-prompt embeddings.

    ``class_prompt_ids`` (K, T) — one tokenised prompt per class (average several templates
    per class before normalising for the CLIP "prompt ensembling" trick).
    """
    v = model.encode_image(images)  # (B, d_e)
    t = model.encode_text(class_prompt_ids)  # (K, d_e)
    return model.logit_scale.exp() * v @ t.T  # (B, K)
