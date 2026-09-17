# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/bert.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k bert -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/bert --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, ids: torch.Tensor, segment_ids: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class MLMHead(nn.Module):
    """Transform + LayerNorm + tied decoder: (B, T, d) -> (B, T, V)."""

    def __init__(self, d_model: int, tok_emb: nn.Embedding) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class NSPHead(nn.Module):
    """Next-sentence prediction from the pooled [CLS]: (B, d) -> (B, 2)."""

    def __init__(self, d_model: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def mask_tokens_for_mlm(ids: torch.Tensor, vocab_size: int, mask_id: int, special_ids: set[int], mask_prob: float=0.15, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT's masking recipe. Of the selected 15% of (non-special) tokens:
        80% -> [MASK], 10% -> random token, 10% -> unchanged.
    Returns (input_ids (B, T), labels (B, T)) with labels = -100 at unselected positions.
    """
    raise NotImplementedError('TODO: implement mask_tokens_for_mlm (see the reference in src/mlbook)')
