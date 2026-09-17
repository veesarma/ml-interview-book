# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/sft.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k sft -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/sft --force

"""Supervised fine-tuning (SFT): assistant-only loss masking, packing, and a training loop.

SFT keeps the pretraining objective (next-token cross-entropy) and changes only
*which tokens are targets*: with ``assistant_only_labels`` the loss is computed on
assistant tokens and the user/system tokens are ignored (label ``-100``).
``pack_examples`` concatenates several short conversations into one row to avoid
padding waste, and ``packed_attention_mask`` builds the block-diagonal causal mask
that stops example ``k`` from attending to example ``k-1``.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlbook.posttrain.toy_lm import causal_mask
IGNORE_INDEX = -100

def assistant_only_labels(input_ids: torch.Tensor, assistant_mask: torch.Tensor) -> torch.Tensor:
    """Labels for next-token prediction with the loss restricted to assistant tokens.

    ``labels[:, t]`` is the target for the logits at position ``t`` (i.e. ``input_ids[:, t+1]``)
    when that next token is an assistant token, else ``IGNORE_INDEX``.

    Args:
        input_ids: (B, T) token ids.
        assistant_mask: (B, T) 1 on tokens that belong to the assistant's turn.
    Returns:
        (B, T) labels; the last column is always ``IGNORE_INDEX`` (nothing follows it).
    """
    raise NotImplementedError('TODO: implement assistant_only_labels (see the reference in src/mlbook)')

def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean autoregressive cross-entropy over non-ignored positions.

    ``L = -(1/|A|) sum_{t in A} log softmax(logits_t)[labels_t]`` with ``A`` = assistant targets.

    Args:
        logits: (B, T, V).
        labels: (B, T) with ``IGNORE_INDEX`` on positions to skip.
    Returns:
        scalar loss.
    """
    raise NotImplementedError('TODO: implement sft_loss (see the reference in src/mlbook)')

@dataclass
class PackedBatch:
    """One packed row-batch: ``input_ids``, ``labels`` and ``segment_ids`` all (B, T)."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    segment_ids: torch.Tensor

def pack_examples(examples: list[tuple[list[int], list[int]]], max_len: int, pad_id: int) -> PackedBatch:
    """Greedily pack ``(input_ids, assistant_mask)`` examples into rows of length ``max_len``.

    Examples are placed in order; a new row starts when the next example does not fit.
    ``segment_ids`` numbers the examples within a row from 1; padding has segment 0.
    Labels are computed per example *before* concatenation, so the first token of an
    example is never a target of the last token of the previous example.

    Returns:
        :class:`PackedBatch` with tensors of shape (B, max_len).
    """
    raise NotImplementedError('TODO: implement pack_examples (see the reference in src/mlbook)')

def packed_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Block-diagonal causal mask: ``allowed[b, i, j] = (j <= i) and seg[b, i] == seg[b, j]``.

    Padding (segment 0) may attend to itself so softmax never sees an all-masked row.

    Args:
        segment_ids: (B, T) ints, 0 for padding.
    Returns:
        (B, T, T) bool.
    """
    raise NotImplementedError('TODO: implement packed_attention_mask (see the reference in src/mlbook)')

def train_sft(model: nn.Module, batches: list[PackedBatch], epochs: int=1, lr: float=0.003, use_packed_mask: bool=True) -> list[float]:
    """Minimal SFT loop: AdamW, constant LR, one pass per epoch over ``batches``.

    Returns:
        list of per-step training losses.
    """
    raise NotImplementedError('TODO: implement train_sft (see the reference in src/mlbook)')
