"""Part VIII — vision Transformers and multimodal models, self-contained.

Dimension vocabulary: ``B`` batch, ``N`` tokens, ``d`` model width, ``H`` heads,
``P`` patch size, ``C, H, W`` image channels/height/width, ``Q`` object queries,
``K`` classes, ``T`` text length or video frames, ``N_v`` visual tokens.
"""

from . import (
    attention_block,
    clip,
    detr_loss,
    hungarian,
    patch_embed,
    projectors,
    swin_window,
    token_compression,
    video_attention,
    vit,
    vlm,
)

__all__ = [
    "attention_block",
    "patch_embed",
    "vit",
    "swin_window",
    "hungarian",
    "detr_loss",
    "clip",
    "projectors",
    "vlm",
    "token_compression",
    "video_attention",
]
