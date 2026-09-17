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

    def __init__(self, d_h: int, d_z: int, d_hidden: int | None = None) -> None:
        super().__init__()
        d_hidden = d_hidden or d_h
        self.fc1 = nn.Linear(d_h, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_z)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc2(F.relu(self.fc1(h)))                        # (B, d_z)
        return z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent over the 2B embeddings ``[z1; z2]``; ``z1[i]`` and ``z2[i]`` are the positive pair.

    Args:
        z1, z2: (B, d_z) projections of the two views (un-normalised; normalised here).
    Returns:
        scalar loss, mean over the 2B anchors.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                               # (2B, d_z)
    z = F.normalize(z, dim=1)                                    # (2B, d_z) unit vectors
    sim = z @ z.t() / temperature                                # (2B, 2B) cosine / τ
    self_mask = torch.eye(2 * B, dtype=torch.bool)               # (2B, 2B)
    sim = sim.masked_fill(self_mask, float("-inf"))              # (2B, 2B) exclude k = i from the softmax
    # positive of anchor i is i + B (first half) or i − B (second half)
    positives = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)])  # (2B,)
    # cross-entropy with the positive as the "class": −log softmax(sim)[i, positives[i]]
    loss = F.cross_entropy(sim, positives)
    return loss


def nt_xent_loss_reference(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Loop version of NT-Xent, written straight from the formula, for testing the vectorised one."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)           # (2B, d_z)
    total = 0.0
    for i in range(2 * B):
        p = i + B if i < B else i - B
        numer = torch.exp(z[i] @ z[p] / temperature)
        denom = sum(torch.exp(z[i] @ z[k] / temperature) for k in range(2 * B) if k != i)
        total = total - torch.log(numer / denom)
    return total / (2 * B)


def info_nce_loss(query: torch.Tensor, key_pos: torch.Tensor, keys_neg: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """MoCo-style InfoNCE with an explicit queue of negatives (one positive per query).

    Args:
        query: (B, d).  key_pos: (B, d).  keys_neg: (Q, d) the queue.
    Returns:
        scalar.
    """
    q = F.normalize(query, dim=1)                                # (B, d)
    k = F.normalize(key_pos, dim=1)                              # (B, d)
    neg = F.normalize(keys_neg, dim=1)                           # (Q, d)
    l_pos = (q * k).sum(dim=1, keepdim=True)                     # (B, 1)
    l_neg = q @ neg.t()                                          # (B, Q)
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature      # (B, 1+Q)
    labels = torch.zeros(q.shape[0], dtype=torch.long)           # (B,) positive is index 0
    return F.cross_entropy(logits, labels)


def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """(B, B) matrix of cosine similarities between the two views (diagonal = positives)."""
    return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).t()  # (B, B)
