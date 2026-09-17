"""FixMatch: pseudo-label from the *weakly* augmented view, train on the *strongly* augmented view.

    q_u = p_θ(· | weak(x_u))              (detached)
    ŷ_u = argmax q_u,  m_u = 1[max q_u ≥ τ]
    L_u = Σ_u m_u · CE( p_θ(· | strong(x_u)), ŷ_u ) / B_u          (note: divided by B_u, not by Σ m_u)
    L   = CE(labelled) + λ_u L_u.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fixmatch_unlabeled_loss(logits_weak: torch.Tensor, logits_strong: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Consistency loss between weak-view pseudo-labels and strong-view predictions.

    Args:
        logits_weak, logits_strong: (B_u, K) from the same unlabelled examples under two augmentations.
    Returns:
        (loss scalar averaged over the full unlabelled batch, fraction of examples above threshold).
    """
    probs_weak = F.softmax(logits_weak.detach(), dim=1)           # (B_u, K)
    conf, pseudo = probs_weak.max(dim=1)                          # (B_u,), (B_u,)
    mask = (conf >= threshold).float()                            # (B_u,)
    ce = F.cross_entropy(logits_strong, pseudo, reduction="none")  # (B_u,)
    loss = (ce * mask).mean()                                     # FixMatch divides by B_u (μB in the paper)
    return loss, mask.mean()


def fixmatch_loss(logits_l: torch.Tensor, y_l: torch.Tensor, logits_weak: torch.Tensor,
                  logits_strong: torch.Tensor, threshold: float = 0.95, lambda_u: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Full FixMatch objective.  Returns (loss, masked fraction)."""
    loss_l = F.cross_entropy(logits_l, y_l)
    loss_u, frac = fixmatch_unlabeled_loss(logits_weak, logits_strong, threshold)
    return loss_l + lambda_u * loss_u, frac


def weak_augment(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Toy 'weak' augmentation for vectors: small Gaussian jitter.  x: (B, d)."""
    return x + sigma * torch.randn_like(x)


def strong_augment(x: torch.Tensor, sigma: float = 0.3, p_drop: float = 0.2) -> torch.Tensor:
    """Toy 'strong' augmentation: large jitter plus random coordinate dropout.  x: (B, d)."""
    keep = (torch.rand_like(x) >= p_drop).to(x.dtype)             # (B, d)
    return (x + sigma * torch.randn_like(x)) * keep


def mean_teacher_consistency(logits_student: torch.Tensor, logits_teacher: torch.Tensor) -> torch.Tensor:
    """Mean Teacher / Π-model consistency: MSE between softmax outputs (teacher detached).  (B, K) each."""
    p_s = F.softmax(logits_student, dim=1)                        # (B, K)
    p_t = F.softmax(logits_teacher.detach(), dim=1)               # (B, K)
    return ((p_s - p_t) ** 2).sum(dim=1).mean()
