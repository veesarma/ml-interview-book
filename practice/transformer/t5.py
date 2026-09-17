# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/t5.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k t5 -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/t5 --force

"""A small T5-style encoder-decoder: shared embedding, pre-RMSNorm blocks, relative position
bias (one learned table per stack, shared by all its layers), cross-attention in every decoder
block, and span-corruption pre-training targets.

    enc = Encoder(x)                                  (B, T_src, d)   bidirectional
    dec = Decoder(y_in, enc)                          (B, T_tgt, d)   causal self-attn + cross-attn(Q=dec, K/V=enc)
    logits = dec E^T / sqrt(d)                        (B, T_tgt, V)   tied, scaled (T5 convention)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .blocks import RMSNorm, TransformerDecoderBlock, TransformerEncoderBlock
from .masks import causal_mask, combine_masks, padding_mask
from .positional import RelativePositionBias

@dataclass
class T5Config:
    vocab_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int | None = None
    dropout: float = 0.0
    rel_buckets: int = 32
    rel_max_distance: int = 128
    pad_id: int = 0
    decoder_start_id: int = 0

class T5(nn.Module):
    """forward(src (B, T_src), tgt_in (B, T_tgt)) -> logits (B, T_tgt, V)."""

    def __init__(self, cfg: T5Config) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src (B, T_src) -> enc (B, T_src, d), src_mask (B, 1, 1, T_src)."""
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, tgt_in: torch.Tensor, enc: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """tgt_in (B, T_tgt), enc (B, T_src, d) -> logits (B, T_tgt, V)."""
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def shift_right(labels: torch.Tensor, decoder_start_id: int) -> torch.Tensor:
    """Decoder inputs for teacher forcing: [start, y_1, ..., y_{T-1}] from labels [y_1, ..., y_T]."""
    raise NotImplementedError('TODO: implement shift_right (see the reference in src/mlbook)')

def span_corruption(tokens: list[int], sentinel_start: int, noise_density: float=0.15, mean_span: float=3.0, rng: np.random.Generator | None=None) -> tuple[list[int], list[int]]:
    """T5's pre-training objective on one sequence.

    Roughly ``noise_density`` of the tokens are dropped in spans of mean length
    ``mean_span``; each dropped span is replaced by one sentinel id in the input and the
    target lists sentinel + dropped tokens for every span, ending with a final sentinel.

        tokens: [a b c d e f g]  ->  input: [a <X> d e <Y> g]   target: [<X> b c <Y> f <Z>]
    """
    raise NotImplementedError('TODO: implement span_corruption (see the reference in src/mlbook)')
