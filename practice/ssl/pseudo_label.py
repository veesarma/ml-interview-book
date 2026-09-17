# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/ssl/pseudo_label.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k pseudo_label -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py ssl/pseudo_label --force

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
    raise NotImplementedError('TODO: implement confident_pseudo_labels (see the reference in src/mlbook)')

def pseudo_label_loss(logits_u: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked cross-entropy of the unlabelled batch against its own confident argmax.

    Args:
        logits_u: (B_u, K) predictions used both for the target (detached) and for the gradient.
    Returns:
        (loss scalar, fraction of the batch above threshold).
    """
    raise NotImplementedError('TODO: implement pseudo_label_loss (see the reference in src/mlbook)')

def semi_supervised_loss(logits_l: torch.Tensor, y_l: torch.Tensor, logits_u: torch.Tensor, threshold: float, lambda_u: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``CE(labelled) + λ_u · pseudo_label_loss(unlabelled)``.  Returns (loss, masked fraction)."""
    raise NotImplementedError('TODO: implement semi_supervised_loss (see the reference in src/mlbook)')

def entropy_minimization_loss(logits_u: torch.Tensor) -> torch.Tensor:
    """``mean_u H(p_θ(·|x_u))``, which pushes predictions on unlabelled data toward one-hot.  logits_u: (B_u, K)."""
    raise NotImplementedError('TODO: implement entropy_minimization_loss (see the reference in src/mlbook)')

def class_balance_of_pseudo_labels(pseudo: torch.Tensor, mask: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Histogram of accepted pseudo-labels (K,).  Watch it to catch confirmation bias and imbalance."""
    raise NotImplementedError('TODO: implement class_balance_of_pseudo_labels (see the reference in src/mlbook)')
