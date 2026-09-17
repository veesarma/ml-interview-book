# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/vlm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k vlm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/vlm --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyCausalLM(nn.Module):
    """Decoder-only LM that accepts input *embeddings*: (B, T, d_llm) -> logits (B, T, V)."""

    def __init__(self, vocab: int, max_len: int, d_llm: int, depth: int, n_heads: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def merge_visual_tokens(text_embeds: torch.Tensor, input_ids: torch.Tensor, visual: torch.Tensor, image_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace the single ``<image>`` slot with N_v visual embeddings.

    ``text_embeds`` (B, T, d), ``input_ids`` (B, T) with exactly one image token per row,
    ``visual`` (B, N_v, d). Returns (merged (B, T-1+N_v, d), position_ids (B, T-1+N_v),
    is_visual (B, T-1+N_v) bool). Every row must have the image token at the same position
    (true after left-aligned prompt templating; general ragged merging is a scatter).
    """
    raise NotImplementedError('TODO: implement merge_visual_tokens (see the reference in src/mlbook)')

class MiniVLM(nn.Module):
    """Vision encoder -> projector -> causal LM with visual-token insertion.

    ``forward(images (B, C, H, W), input_ids (B, T))`` -> logits (B, T-1+N_v, V) and the
    ``is_visual`` mask (B, T-1+N_v) so the caller can drop visual positions from the loss.
    """

    def __init__(self, vision: ToyVisionEncoder, d_v: int, lm: TinyCausalLM, d_llm: int, image_token_id: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def vlm_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, is_visual: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Next-token loss on text positions only.

    ``logits`` (B, L, V) over the merged sequence; ``input_ids`` (B, T) original text with one
    image token; ``is_visual`` (B, L). Targets are built by expanding the image slot with
    ``ignore_index`` (-100) so no loss is computed on predicting visual tokens.
    """
    raise NotImplementedError('TODO: implement vlm_lm_loss (see the reference in src/mlbook)')
