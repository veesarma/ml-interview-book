"""A small BERT: bidirectional Post-LN encoder, masked-language-model head, 80/10/10 masking.

    input  = TokEmb[x] + PosEmb[0..T-1] + SegEmb[segment]      (B, T, d)
    h      = EncoderBlock_L(... EncoderBlock_1(input))          no causal mask; pads masked as keys
    MLM    : logits = softmax over V at the masked positions only
    NSP    : binary classifier on the pooled [CLS] vector (dropped by RoBERTa)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import TransformerEncoderBlock
from .masks import padding_mask


@dataclass
class BERTConfig:
    vocab_size: int
    max_len: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int | None = None
    dropout: float = 0.0
    n_segments: int = 2
    pad_id: int = 0


class BERT(nn.Module):
    """forward(ids (B, T), segment_ids (B, T) | None) -> (hidden (B, T, d), pooled (B, d))."""

    def __init__(self, cfg: BERTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)  # (V, d)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)  # (T_max, d)
        self.seg_emb = nn.Embedding(cfg.n_segments, cfg.d_model)  # (2, d)
        self.emb_norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, norm="layer", pre_norm=False) for _ in range(cfg.n_layers)]
        )
        self.pooler = nn.Linear(cfg.d_model, cfg.d_model)  # applied to the [CLS] position

    def forward(self, ids: torch.Tensor, segment_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = ids.shape
        if segment_ids is None:
            segment_ids = torch.zeros_like(ids)  # (B, T)
        pos = torch.arange(T, device=ids.device)[None, :]  # (1, T)
        x = self.tok_emb(ids) + self.pos_emb(pos) + self.seg_emb(segment_ids)  # (B, T, d)
        x = self.drop(self.emb_norm(x))
        mask = padding_mask(ids == self.cfg.pad_id)  # (B, 1, 1, T) keys that are real tokens
        for block in self.blocks:
            x = block(x, mask)  # (B, T, d) every position sees every non-pad position
        pooled = torch.tanh(self.pooler(x[:, 0, :]))  # (B, d) from [CLS] at position 0
        return x, pooled


class MLMHead(nn.Module):
    """Transform + LayerNorm + tied decoder: (B, T, d) -> (B, T, V)."""

    def __init__(self, d_model: int, tok_emb: nn.Embedding) -> None:
        super().__init__()
        self.transform = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, tok_emb.num_embeddings, bias=False)  # (d, V)
        self.decoder.weight = tok_emb.weight  # tied to the input embedding
        self.bias = nn.Parameter(torch.zeros(tok_emb.num_embeddings))  # (V,) separate output bias

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.norm(F.gelu(self.transform(hidden)))  # (B, T, d)
        return self.decoder(h) + self.bias  # (B, T, V)


class NSPHead(nn.Module):
    """Next-sentence prediction from the pooled [CLS]: (B, d) -> (B, 2)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)  # (B, 2)


def mask_tokens_for_mlm(
    ids: torch.Tensor,
    vocab_size: int,
    mask_id: int,
    special_ids: set[int],
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT's masking recipe. Of the selected 15% of (non-special) tokens:
        80% -> [MASK], 10% -> random token, 10% -> unchanged.
    Returns (input_ids (B, T), labels (B, T)) with labels = -100 at unselected positions.
    """
    labels = ids.clone()  # (B, T)
    eligible = torch.ones_like(ids, dtype=torch.bool)  # (B, T)
    for sid in special_ids:
        eligible &= ids != sid
    selected = (torch.rand(ids.shape, generator=generator) < mask_prob) & eligible  # (B, T)
    labels[~selected] = -100
    inputs = ids.clone()  # (B, T)
    u = torch.rand(ids.shape, generator=generator)  # (B, T)
    to_mask = selected & (u < 0.8)  # 80%
    to_random = selected & (u >= 0.8) & (u < 0.9)  # 10%
    inputs[to_mask] = mask_id
    random_tokens = torch.randint(0, vocab_size, ids.shape, generator=generator)  # (B, T)
    inputs[to_random] = random_tokens[to_random]
    return inputs, labels  # remaining 10% keep their original id (but are still predicted)
