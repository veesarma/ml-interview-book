"""Pseudo-labelling / self-training: use a model's confident predictions on unlabelled data as targets.

    ŷ_u = argmax_k p_θ(k | x_u),   m_u = 1[max_k p_θ(k | x_u) ≥ τ],
    L = CE(labelled) + λ_u · Σ_u m_u CE(p_θ(·|x_u), ŷ_u) / max(Σ_u m_u, 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def confident_pseudo_labels(logits_u: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard pseudo-labels and a confidence mask.

    Args:
        logits_u: (B_u, K).
    Returns:
        pseudo: (B_u,) argmax class;  mask: (B_u,) float 1.0 where max prob ≥ threshold.
    """
    probs = F.softmax(logits_u.detach(), dim=1)                   # (B_u, K)  targets carry no gradient
    conf, pseudo = probs.max(dim=1)                               # (B_u,), (B_u,)
    mask = (conf >= threshold).float()                            # (B_u,)
    return pseudo, mask


def pseudo_label_loss(logits_u: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked cross-entropy of the unlabelled batch against its own confident argmax.

    Args:
        logits_u: (B_u, K) predictions used both for the target (detached) and for the gradient.
    Returns:
        (loss scalar, fraction of the batch above threshold).
    """
    pseudo, mask = confident_pseudo_labels(logits_u, threshold)   # (B_u,), (B_u,)
    ce = F.cross_entropy(logits_u, pseudo, reduction="none")      # (B_u,)
    loss = (ce * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, mask.mean()


def semi_supervised_loss(logits_l: torch.Tensor, y_l: torch.Tensor, logits_u: torch.Tensor,
                         threshold: float, lambda_u: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``CE(labelled) + λ_u · pseudo_label_loss(unlabelled)``.  Returns (loss, masked fraction)."""
    loss_l = F.cross_entropy(logits_l, y_l)
    loss_u, frac = pseudo_label_loss(logits_u, threshold)
    return loss_l + lambda_u * loss_u, frac


def entropy_minimization_loss(logits_u: torch.Tensor) -> torch.Tensor:
    """``mean_u H(p_θ(·|x_u))`` — pushes predictions on unlabelled data toward one-hot.  logits_u: (B_u, K)."""
    log_p = F.log_softmax(logits_u, dim=1)                        # (B_u, K)
    return -(log_p.exp() * log_p).sum(dim=1).mean()


def class_balance_of_pseudo_labels(pseudo: torch.Tensor, mask: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Histogram of accepted pseudo-labels (K,) — watch this to catch confirmation bias / imbalance."""
    accepted = pseudo[mask > 0.5]                                 # (M,)
    return torch.bincount(accepted, minlength=n_classes).float()  # (K,)
