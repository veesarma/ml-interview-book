"""Sequence packing and document-aware attention masks.

Pretraining batches are fixed-length rows of ``T`` tokens.  Documents have
arbitrary lengths, so we *pack* several into one row separated by EOS and
record a ``doc_id`` per position.  The attention mask is then

    allowed[i, j] = (j <= i) and (doc_id[i] == doc_id[j])

so a token never attends across a document boundary (Llama 3 masks this way;
GPT-3 style training lets attention cross the EOS and relies on the model
learning to ignore it).
"""

from __future__ import annotations

import torch


def pack_sequences(
    docs: list[list[int]], max_len: int, eos_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Greedy first-fit packing of tokenised documents into rows of ≤ ``max_len``.

    Each document gets an EOS appended, then is placed into the first row with
    room; documents longer than ``max_len`` are split across consecutive rows.

    Args:
        docs: list of token-id lists (variable length).
        max_len: row length ``T``.
        eos_id: separator token id.
    Returns:
        (rows, doc_ids): ``rows[r]`` is a list of ≤ T token ids, ``doc_ids[r]`` the
        parallel list of document indices (0-based, one per token).
    """
    rows: list[list[int]] = []
    doc_ids: list[list[int]] = []
    for d_idx, doc in enumerate(docs):
        remaining = list(doc) + [eos_id]
        while remaining:
            # first-fit: the first row with any free space, else a new row
            target = next((r for r in range(len(rows)) if len(rows[r]) < max_len), None)
            if target is None:
                rows.append([])
                doc_ids.append([])
                target = len(rows) - 1
            room = max_len - len(rows[target])
            chunk, remaining = remaining[:room], remaining[room:]
            rows[target].extend(chunk)
            doc_ids[target].extend([d_idx] * len(chunk))
    return rows, doc_ids


def pad_rows(rows: list[list[int]], max_len: int, pad_id: int) -> torch.Tensor:
    """Right-pad packed rows into a ``(B, T)`` long tensor."""
    out = torch.full((len(rows), max_len), pad_id, dtype=torch.long)  # (B, T)
    for r, row in enumerate(rows):
        out[r, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def document_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Boolean attention mask that is causal *and* blocks cross-document attention.

    Args:
        doc_ids: (B, T) integer document id per position (use -1 for padding).
    Returns:
        (B, T, T) bool, ``True`` where query i may attend key j.
    """
    B, T = doc_ids.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))  # (T, T)
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]  # (B, T, T)
    return causal[None, :, :] & same_doc  # (B, T, T)


def position_ids_within_document(doc_ids: torch.Tensor) -> torch.Tensor:
    """Per-position index that restarts at 0 at every document boundary.

    Args:
        doc_ids: (B, T) document ids.
    Returns:
        (B, T) long tensor of positions inside each document.
    """
    B, T = doc_ids.shape
    pos = torch.zeros(B, T, dtype=torch.long)  # (B, T)
    for b in range(B):
        for t in range(1, T):
            pos[b, t] = pos[b, t - 1] + 1 if doc_ids[b, t] == doc_ids[b, t - 1] else 0
    return pos
