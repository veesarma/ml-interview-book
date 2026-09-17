# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/packing.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k packing -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/packing --force

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

def pack_sequences(docs: list[list[int]], max_len: int, eos_id: int) -> tuple[list[list[int]], list[list[int]]]:
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
    raise NotImplementedError('TODO: implement pack_sequences (see the reference in src/mlbook)')

def pad_rows(rows: list[list[int]], max_len: int, pad_id: int) -> torch.Tensor:
    """Right-pad packed rows into a ``(B, T)`` long tensor."""
    raise NotImplementedError('TODO: implement pad_rows (see the reference in src/mlbook)')

def document_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Boolean attention mask that is causal *and* blocks cross-document attention.

    Args:
        doc_ids: (B, T) integer document id per position (use -1 for padding).
    Returns:
        (B, T, T) bool, ``True`` where query i may attend key j.
    """
    raise NotImplementedError('TODO: implement document_causal_mask (see the reference in src/mlbook)')

def position_ids_within_document(doc_ids: torch.Tensor) -> torch.Tensor:
    """Per-position index that restarts at 0 at every document boundary.

    Args:
        doc_ids: (B, T) document ids.
    Returns:
        (B, T) long tensor of positions inside each document.
    """
    raise NotImplementedError('TODO: implement position_ids_within_document (see the reference in src/mlbook)')
