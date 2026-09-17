"""A mini vision-language model in the LLaVA pattern:

    image (B, C, H, W) --vision encoder--> (B, N_v, d_v) --projector--> (B, N_v, d_llm)
    text ids (B, T) --embedding--> (B, T, d_llm)
    sequence = [text before <image>] ++ [visual tokens] ++ [text after]   (B, T - 1 + N_v, d_llm)
    causal LM over the merged sequence; loss only on text positions after the image.

The image placeholder token ``image_token_id`` occupies exactly one slot in the
input ids and is *expanded* to N_v visual embeddings.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention_block import TransformerBlock
from .patch_embed import PatchEmbedLinear
from .projectors import MLPProjector


class ToyVisionEncoder(nn.Module):
    """Patch-embed + Transformer blocks, returns ALL patch tokens: (B, C, H, W) -> (B, N_v, d_v)."""

    def __init__(self, image_size: int, patch: int, in_channels: int, d_v: int, depth: int, n_heads: int) -> None:
        super().__init__()
        n = (image_size // patch) ** 2
        self.embed = PatchEmbedLinear(in_channels, patch, d_v)
        self.pos = nn.Parameter(torch.zeros(1, n, d_v))  # (1, N_v, d_v)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_v, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.embed(x) + self.pos  # (B, N_v, d_v)
        for blk in self.blocks:
            t = blk(t)  # (B, N_v, d_v)
        return self.norm(t)  # (B, N_v, d_v)


class TinyCausalLM(nn.Module):
    """Decoder-only LM that accepts input *embeddings*: (B, T, d_llm) -> logits (B, T, V)."""

    def __init__(self, vocab: int, max_len: int, d_llm: int, depth: int, n_heads: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_llm)
        self.pos = nn.Embedding(max_len, d_llm)
        self.blocks = nn.ModuleList([TransformerBlock(d_llm, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_llm)
        self.head = nn.Linear(d_llm, vocab, bias=False)

    def forward(self, embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = embeds.shape
        x = embeds + self.pos(position_ids)  # (B, T, d_llm)
        causal = torch.full((T, T), -1e9, device=embeds.device).triu(1)  # (T, T): 0 on/below diag
        for blk in self.blocks:
            x = blk(x, causal[None, None])  # (B, T, d_llm)
        return self.head(self.norm(x))  # (B, T, V)


def merge_visual_tokens(
    text_embeds: torch.Tensor, input_ids: torch.Tensor, visual: torch.Tensor, image_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace the single ``<image>`` slot with N_v visual embeddings.

    ``text_embeds`` (B, T, d), ``input_ids`` (B, T) with exactly one image token per row,
    ``visual`` (B, N_v, d). Returns (merged (B, T-1+N_v, d), position_ids (B, T-1+N_v),
    is_visual (B, T-1+N_v) bool). Every row must have the image token at the same position
    (true after left-aligned prompt templating; general ragged merging is a scatter).
    """
    B, T, d = text_embeds.shape
    pos = (input_ids == image_token_id).nonzero()  # (B, 2): (row, col)
    if pos.shape[0] != B or not bool((pos[:, 1] == pos[0, 1]).all()):
        raise ValueError("expected exactly one <image> token per row, at the same column")
    p = int(pos[0, 1])
    N_v = visual.shape[1]
    merged = torch.cat([text_embeds[:, :p], visual, text_embeds[:, p + 1 :]], dim=1)  # (B, T-1+N_v, d)
    L = merged.shape[1]
    position_ids = torch.arange(L, device=text_embeds.device)[None].expand(B, -1)  # (B, L)
    is_visual = torch.zeros(B, L, dtype=torch.bool, device=text_embeds.device)  # (B, L)
    is_visual[:, p : p + N_v] = True
    return merged, position_ids, is_visual


class MiniVLM(nn.Module):
    """Vision encoder -> projector -> causal LM with visual-token insertion.

    ``forward(images (B, C, H, W), input_ids (B, T))`` -> logits (B, T-1+N_v, V) and the
    ``is_visual`` mask (B, T-1+N_v) so the caller can drop visual positions from the loss.
    """

    def __init__(self, vision: ToyVisionEncoder, d_v: int, lm: TinyCausalLM, d_llm: int, image_token_id: int) -> None:
        super().__init__()
        self.vision = vision
        self.projector = MLPProjector(d_v, d_llm)
        self.lm = lm
        self.image_token_id = image_token_id

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.vision(images)  # (B, N_v, d_v)
        visual = self.projector(feats)  # (B, N_v, d_llm)
        text = self.lm.tok(input_ids)  # (B, T, d_llm)
        merged, position_ids, is_visual = merge_visual_tokens(text, input_ids, visual, self.image_token_id)  # (B, L, d_llm)
        logits = self.lm(merged, position_ids)  # (B, L, V)
        return logits, is_visual


def vlm_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, is_visual: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Next-token loss on text positions only.

    ``logits`` (B, L, V) over the merged sequence; ``input_ids`` (B, T) original text with one
    image token; ``is_visual`` (B, L). Targets are built by expanding the image slot with
    ``ignore_index`` (-100) so no loss is computed on predicting visual tokens.
    """
    B, L, V = logits.shape
    N_v = int(is_visual[0].sum())
    p = int(is_visual[0].nonzero()[0, 0])
    ignore = torch.full((B, N_v), -100, dtype=torch.long, device=input_ids.device)  # (B, N_v)
    targets = torch.cat([input_ids[:, :p], ignore, input_ids[:, p + 1 :]], dim=1)  # (B, L)
    pred = logits[:, :-1].reshape(-1, V)  # (B*(L-1), V)
    tgt = targets[:, 1:].reshape(-1)  # (B*(L-1),)
    return nn.functional.cross_entropy(pred, tgt, ignore_index=-100)
