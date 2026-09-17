# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/ssl/simclr.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k simclr -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py ssl/simclr --force

"""SimCLR: NT-Xent (normalised-temperature cross-entropy) contrastive loss and the projection head.

For a batch of B images augmented twice we get 2B embeddings.  For anchor i with positive p(i):

    ℓ_i = −log  exp(sim(z_i, z_{p(i)}) / τ)  /  Σ_{k ≠ i} exp(sim(z_i, z_k) / τ),

sim = cosine similarity, and the loss is the mean of ℓ_i over all 2B anchors.  Each anchor has
one positive and 2B − 2 negatives; the denominator excludes only the anchor itself.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class ProjectionHead(nn.Module):
    """``g(h) = W₂ ReLU(W₁ h)``: maps encoder features (B, d_h) to contrastive space (B, d_z)."""

    def __init__(self, d_h: int, d_z: int, d_hidden: int | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float=0.1) -> torch.Tensor:
    """NT-Xent over the 2B embeddings ``[z1; z2]``; ``z1[i]`` and ``z2[i]`` are the positive pair.

    Args:
        z1, z2: (B, d_z) projections of the two views (un-normalised; normalised here).
    Returns:
        scalar loss, mean over the 2B anchors.
    """
    raise NotImplementedError('TODO: implement nt_xent_loss (see the reference in src/mlbook)')

def nt_xent_loss_reference(z1: torch.Tensor, z2: torch.Tensor, temperature: float=0.1) -> torch.Tensor:
    """Loop version of NT-Xent, written straight from the formula, for testing the vectorised one."""
    raise NotImplementedError('TODO: implement nt_xent_loss_reference (see the reference in src/mlbook)')

def info_nce_loss(query: torch.Tensor, key_pos: torch.Tensor, keys_neg: torch.Tensor, temperature: float=0.07) -> torch.Tensor:
    """MoCo-style InfoNCE with an explicit queue of negatives (one positive per query).

    Args:
        query: (B, d).  key_pos: (B, d).  keys_neg: (Q, d) the queue.
    Returns:
        scalar.
    """
    raise NotImplementedError('TODO: implement info_nce_loss (see the reference in src/mlbook)')

def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """(B, B) matrix of cosine similarities between the two views (diagonal = positives)."""
    raise NotImplementedError('TODO: implement cosine_similarity_matrix (see the reference in src/mlbook)')
