# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/detection/focal_loss.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k focal_loss -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py detection/focal_loss --force

"""Focal loss (Lin et al. 2017) in PyTorch, plus a hand-derived NumPy gradient for testing.

For one anchor with logit ``z``, ``p = σ(z)``, label ``y ∈ {0, 1}``, and ``p_t = p`` if ``y=1``
else ``1 − p``:

    FL = −α_t (1 − p_t)^γ log p_t.

``(1 − p_t)^γ`` down-weights easy examples (``p_t → 1``); ``α_t`` rebalances the classes.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float=0.25, gamma: float=2.0, reduction: str='sum') -> torch.Tensor:
    """Binary focal loss on raw logits.

    Args:
        logits: (N, K) or any shape.  targets: same shape, values in {0, 1} (float).
    Returns:
        scalar (``sum``/``mean``) or per-element loss (``none``).
    """
    raise NotImplementedError('TODO: implement sigmoid_focal_loss (see the reference in src/mlbook)')

def focal_loss_grad_numpy(z: np.ndarray, y: np.ndarray, alpha: float=0.25, gamma: float=2.0) -> np.ndarray:
    """``∂FL/∂z`` derived by hand, for the positive and negative cases.

    Positive (y = 1), p = σ(z):
        FL = −α (1−p)^γ log p
        dFL/dz = α (1−p)^γ · [ γ p log p − (1−p) ]        (using dp/dz = p(1−p))
    Negative (y = 0):
        FL = −(1−α) p^γ log(1−p)
        dFL/dz = (1−α) p^γ · [ p − γ (1−p) log(1−p) ]
    Note the γ p log p term: for easy positives (p→1) both terms vanish, which is the point.
    """
    raise NotImplementedError('TODO: implement focal_loss_grad_numpy (see the reference in src/mlbook)')

def prior_bias_init(prior_prob: float=0.01) -> float:
    """Bias for the classification conv so every anchor starts with ``σ(b) = π``: ``b = −log((1−π)/π)``.

    With ~100k anchors and ~0.01 prior, the initial loss is dominated by the few positives
    instead of the sea of confident-wrong negatives, which kept RetinaNet training stable.
    """
    raise NotImplementedError('TODO: implement prior_bias_init (see the reference in src/mlbook)')
