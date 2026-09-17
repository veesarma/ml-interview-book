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
    decoder_start_id: int = 0  # T5 starts decoding from the pad token


class T5(nn.Module):
    """forward(src (B, T_src), tgt_in (B, T_tgt)) -> logits (B, T_tgt, V)."""

    def __init__(self, cfg: T5Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.shared = nn.Embedding(cfg.vocab_size, cfg.d_model)  # (V, d) used by encoder, decoder and head
        self.enc_bias = RelativePositionBias(cfg.n_heads, cfg.rel_buckets, cfg.rel_max_distance, bidirectional=True)
        self.dec_bias = RelativePositionBias(cfg.n_heads, cfg.rel_buckets, cfg.rel_max_distance, bidirectional=False)
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, norm="rms", pre_norm=True, ffn="gelu") for _ in range(cfg.n_layers)]
        )
        self.decoder = nn.ModuleList(
            [TransformerDecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, norm="rms", pre_norm=True, ffn="gelu", cross_attention=True) for _ in range(cfg.n_layers)]
        )
        self.enc_final = RMSNorm(cfg.d_model)
        self.dec_final = RMSNorm(cfg.d_model)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src (B, T_src) -> enc (B, T_src, d), src_mask (B, 1, 1, T_src)."""
        T = src.shape[1]
        x = self.shared(src)  # (B, T_src, d)
        src_mask = padding_mask(src == self.cfg.pad_id)  # (B, 1, 1, T_src)
        bias = self.enc_bias(T, T)  # (1, H, T_src, T_src)
        for block in self.encoder:
            x = block(x, src_mask, bias)  # (B, T_src, d)
        return self.enc_final(x), src_mask

    def decode(self, tgt_in: torch.Tensor, enc: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """tgt_in (B, T_tgt), enc (B, T_src, d) -> logits (B, T_tgt, V)."""
        T = tgt_in.shape[1]
        y = self.shared(tgt_in)  # (B, T_tgt, d)
        self_mask = combine_masks(causal_mask(T, tgt_in.device), padding_mask(tgt_in == self.cfg.pad_id))  # (B, 1, T_tgt, T_tgt)
        bias = self.dec_bias(T, T)  # (1, H, T_tgt, T_tgt)
        for block in self.decoder:
            y = block(y, enc, self_mask, src_mask, None, None, bias)  # (B, T_tgt, d)
        y = self.dec_final(y)  # (B, T_tgt, d)
        return (y @ self.shared.weight.T) * (self.cfg.d_model ** -0.5)  # (B, T_tgt, V) tied + scaled

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        enc, src_mask = self.encode(src)
        return self.decode(tgt_in, enc, src_mask)


def shift_right(labels: torch.Tensor, decoder_start_id: int) -> torch.Tensor:
    """Decoder inputs for teacher forcing: [start, y_1, ..., y_{T-1}] from labels [y_1, ..., y_T]."""
    start = torch.full_like(labels[:, :1], decoder_start_id)  # (B, 1)
    return torch.cat([start, labels[:, :-1]], dim=1)  # (B, T)


def span_corruption(tokens: list[int], sentinel_start: int, noise_density: float = 0.15, mean_span: float = 3.0, rng: np.random.Generator | None = None) -> tuple[list[int], list[int]]:
    """T5's pre-training objective on one sequence.

    Roughly ``noise_density`` of the tokens are dropped in spans of mean length
    ``mean_span``; each dropped span is replaced by one sentinel id in the input and the
    target lists sentinel + dropped tokens for every span, ending with a final sentinel.

        tokens: [a b c d e f g]  ->  input: [a <X> d e <Y> g]   target: [<X> b c <Y> f <Z>]
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(tokens)
    n_noise = max(1, int(round(n * noise_density)))
    n_spans = max(1, int(round(n_noise / mean_span)))
    is_noise = np.zeros(n, dtype=bool)  # (n,)
    starts = sorted(rng.choice(n, size=min(n_spans, n), replace=False).tolist())
    per_span = max(1, n_noise // n_spans)
    for s in starts:
        is_noise[s : s + per_span] = True
    inputs, targets = [], []
    sentinel = sentinel_start
    i = 0
    while i < n:
        if not is_noise[i]:
            inputs.append(tokens[i])
            i += 1
            continue
        inputs.append(sentinel)
        targets.append(sentinel)
        while i < n and is_noise[i]:
            targets.append(tokens[i])
            i += 1
        sentinel += 1
    targets.append(sentinel)  # closing sentinel
    return inputs, targets
