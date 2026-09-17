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
    labels = torch.full_like(input_ids, IGNORE_INDEX)  # (B, T)
    next_ids = input_ids[:, 1:]  # (B, T-1) the token after each position
    next_is_assistant = assistant_mask[:, 1:].bool()  # (B, T-1)
    labels[:, :-1] = torch.where(next_is_assistant, next_ids, torch.full_like(next_ids, IGNORE_INDEX))
    return labels  # (B, T)


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean autoregressive cross-entropy over non-ignored positions.

    ``L = -(1/|A|) sum_{t in A} log softmax(logits_t)[labels_t]`` with ``A`` = assistant targets.

    Args:
        logits: (B, T, V).
        labels: (B, T) with ``IGNORE_INDEX`` on positions to skip.
    Returns:
        scalar loss.
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)  # (B*T, V)
    flat_labels = labels.reshape(B * T)  # (B*T,)
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX)  # scalar


@dataclass
class PackedBatch:
    """One packed row-batch: ``input_ids``, ``labels`` and ``segment_ids`` all (B, T)."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    segment_ids: torch.Tensor


def pack_examples(
    examples: list[tuple[list[int], list[int]]], max_len: int, pad_id: int
) -> PackedBatch:
    """Greedily pack ``(input_ids, assistant_mask)`` examples into rows of length ``max_len``.

    Examples are placed in order; a new row starts when the next example does not fit.
    ``segment_ids`` numbers the examples within a row from 1; padding has segment 0.
    Labels are computed per example *before* concatenation, so the first token of an
    example is never a target of the last token of the previous example.

    Returns:
        :class:`PackedBatch` with tensors of shape (B, max_len).
    """
    rows: list[list[int]] = [[]]
    row_labels: list[list[int]] = [[]]
    row_segments: list[list[int]] = [[]]
    for ids, amask in examples:
        if len(ids) > max_len:
            raise ValueError("example longer than max_len; truncate first")
        if len(rows[-1]) + len(ids) > max_len:
            rows.append([]), row_labels.append([]), row_segments.append([])
        ids_t = torch.tensor([ids])  # (1, T_ex)
        amask_t = torch.tensor([amask])  # (1, T_ex)
        labels = assistant_only_labels(ids_t, amask_t)[0].tolist()  # (T_ex,)
        seg = max(row_segments[-1], default=0) + 1
        rows[-1].extend(ids), row_labels[-1].extend(labels), row_segments[-1].extend([seg] * len(ids))
    B = len(rows)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)  # (B, T)
    labels_out = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)  # (B, T)
    segment_ids = torch.zeros((B, max_len), dtype=torch.long)  # (B, T)
    for b in range(B):
        n = len(rows[b])
        input_ids[b, :n] = torch.tensor(rows[b])
        labels_out[b, :n] = torch.tensor(row_labels[b])
        segment_ids[b, :n] = torch.tensor(row_segments[b])
    return PackedBatch(input_ids, labels_out, segment_ids)


def packed_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Block-diagonal causal mask: ``allowed[b, i, j] = (j <= i) and seg[b, i] == seg[b, j]``.

    Padding (segment 0) may attend to itself so softmax never sees an all-masked row.

    Args:
        segment_ids: (B, T) ints, 0 for padding.
    Returns:
        (B, T, T) bool.
    """
    B, T = segment_ids.shape
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)  # (B, T, T)
    causal = causal_mask(T, segment_ids.device).unsqueeze(0)  # (1, T, T)
    return same_segment & causal  # (B, T, T)


def train_sft(
    model: nn.Module,
    batches: list[PackedBatch],
    epochs: int = 1,
    lr: float = 3e-3,
    use_packed_mask: bool = True,
) -> list[float]:
    """Minimal SFT loop: AdamW, constant LR, one pass per epoch over ``batches``.

    Returns:
        list of per-step training losses.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        for batch in batches:
            allowed = packed_attention_mask(batch.segment_ids) if use_packed_mask else None  # (B, T, T)
            logits = model(batch.input_ids, allowed)  # (B, T, V)
            loss = sft_loss(logits, batch.labels)  # scalar
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    return losses
