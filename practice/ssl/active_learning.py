# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/ssl/active_learning.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k active_learning -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py ssl/active_learning --force

"""Active-learning acquisition: uncertainty (least confidence, margin, entropy), diversity
(k-center greedy / core-set) and BADGE (k-means++ on gradient embeddings).
"""
from __future__ import annotations
import numpy as np

def least_confidence(probs: np.ndarray) -> np.ndarray:
    """``1 − max_k p_k``.  probs: (N, K) → (N,); larger = more uncertain."""
    raise NotImplementedError('TODO: implement least_confidence (see the reference in src/mlbook)')

def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    """``−(p_(1) − p_(2))`` (negative top-two margin).  probs: (N, K) → (N,)."""
    raise NotImplementedError('TODO: implement margin_uncertainty (see the reference in src/mlbook)')

def entropy_uncertainty(probs: np.ndarray) -> np.ndarray:
    """``H(p) = −Σ_k p_k log p_k``.  probs: (N, K) → (N,)."""
    raise NotImplementedError('TODO: implement entropy_uncertainty (see the reference in src/mlbook)')

def k_center_greedy(features: np.ndarray, labeled_idx: np.ndarray, budget: int) -> np.ndarray:
    """Core-set selection: repeatedly pick the point farthest from the current labelled set.

    Args:
        features: (N, d).  labeled_idx: (n_l,) indices already labelled.
    Returns:
        (budget,) indices of the new points to label.
    """
    raise NotImplementedError('TODO: implement k_center_greedy (see the reference in src/mlbook)')

def badge_gradient_embeddings(probs: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Gradient of the CE loss w.r.t. the last linear layer, using the argmax as pseudo-label.

    For last layer logits = W h, the gradient w.r.t. W is (p − e_ŷ) ⊗ h, flattened to (N, K·d).
    Its norm is large when the model is uncertain (p far from one-hot), which is uncertainty for free,
    and its direction depends on h, which is diversity for free.
    """
    raise NotImplementedError('TODO: implement badge_gradient_embeddings (see the reference in src/mlbook)')

def kmeans_plus_plus_seeding(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialisation: sample points with probability ∝ squared distance to chosen set.

    Args:
        x: (N, d).
    Returns:
        (k,) chosen indices.
    """
    raise NotImplementedError('TODO: implement kmeans_plus_plus_seeding (see the reference in src/mlbook)')

def badge_select(probs: np.ndarray, features: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """BADGE: k-means++ seeding on the gradient embeddings.  Returns (budget,) indices."""
    raise NotImplementedError('TODO: implement badge_select (see the reference in src/mlbook)')
