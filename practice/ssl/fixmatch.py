# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/ssl/fixmatch.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k fixmatch -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py ssl/fixmatch --force

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
    raise NotImplementedError('TODO: implement fixmatch_unlabeled_loss (see the reference in src/mlbook)')

def fixmatch_loss(logits_l: torch.Tensor, y_l: torch.Tensor, logits_weak: torch.Tensor, logits_strong: torch.Tensor, threshold: float=0.95, lambda_u: float=1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Full FixMatch objective.  Returns (loss, masked fraction)."""
    raise NotImplementedError('TODO: implement fixmatch_loss (see the reference in src/mlbook)')

def weak_augment(x: torch.Tensor, sigma: float=0.05) -> torch.Tensor:
    """Toy 'weak' augmentation for vectors: small Gaussian jitter.  x: (B, d)."""
    raise NotImplementedError('TODO: implement weak_augment (see the reference in src/mlbook)')

def strong_augment(x: torch.Tensor, sigma: float=0.3, p_drop: float=0.2) -> torch.Tensor:
    """Toy 'strong' augmentation: large jitter plus random coordinate dropout.  x: (B, d)."""
    raise NotImplementedError('TODO: implement strong_augment (see the reference in src/mlbook)')

def mean_teacher_consistency(logits_student: torch.Tensor, logits_teacher: torch.Tensor) -> torch.Tensor:
    """Mean Teacher / Π-model consistency: MSE between softmax outputs (teacher detached).  (B, K) each."""
    raise NotImplementedError('TODO: implement mean_teacher_consistency (see the reference in src/mlbook)')
