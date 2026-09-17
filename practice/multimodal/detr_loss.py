# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/detr_loss.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k detr_loss -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/detr_loss --force

"""DETR-style set prediction: matching cost, Hungarian matcher, and the loss.

Predictions: class logits ``(B, Q, K+1)`` (last index = "no object") and boxes
``(B, Q, 4)`` in normalised ``(cx, cy, w, h)``. Targets: a list of dicts with
``labels`` ``(T_b,)`` and ``boxes`` ``(T_b, 4)``.

Matching cost for prediction q and target t:
    C[q, t] = -p_q(c_t) + λ_L1 ||b_q - b_t||_1 + λ_giou (-GIoU(b_q, b_t))
Loss after matching σ:
    L = CE(logits, labels ∪ {no-object}, w_noobj) + λ_1 Σ L1 + λ_2 Σ (1 - GIoU)
"""
from __future__ import annotations
import numpy as np
import torch
from .hungarian import hungarian

def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """(..., 4) (cx, cy, w, h) -> (..., 4) (x1, y1, x2, y2)."""
    raise NotImplementedError('TODO: implement box_cxcywh_to_xyxy (see the reference in src/mlbook)')

def pairwise_giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalised IoU between every pair: ``a`` (N, 4), ``b`` (M, 4), xyxy -> (N, M).

    GIoU = IoU - |C \\ (A ∪ B)| / |C|, C = smallest enclosing box. Range (-1, 1].
    """
    raise NotImplementedError('TODO: implement pairwise_giou (see the reference in src/mlbook)')

@torch.no_grad()
def hungarian_match(logits: torch.Tensor, boxes: torch.Tensor, targets: list[dict], w_class: float=1.0, w_l1: float=5.0, w_giou: float=2.0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per image, the (pred_idx, tgt_idx) pairs minimising the matching cost.

    ``logits`` (B, Q, K+1), ``boxes`` (B, Q, 4) cxcywh. Returns a list of B tuples of long tensors (T_b,).
    """
    raise NotImplementedError('TODO: implement hungarian_match (see the reference in src/mlbook)')

def detr_loss(logits: torch.Tensor, boxes: torch.Tensor, targets: list[dict], w_l1: float=5.0, w_giou: float=2.0, eos_coef: float=0.1) -> dict[str, torch.Tensor]:
    """Set loss L = CE + w_l1 * L1 + w_giou * (1 - GIoU), box terms normalised by number of targets.

    ``logits`` (B, Q, K+1), ``boxes`` (B, Q, 4). Unmatched queries are trained to
    predict "no object" (class index K) with weight ``eos_coef`` in the CE.
    Returns dict with ``loss``, ``loss_ce``, ``loss_l1``, ``loss_giou``.
    """
    raise NotImplementedError('TODO: implement detr_loss (see the reference in src/mlbook)')

def matching_as_permutation(match: list[tuple[torch.Tensor, torch.Tensor]], Q: int) -> list[np.ndarray]:
    """For tests: turn each image's (rows, cols) into an array ``perm`` (Q,) with perm[q] = target index or -1."""
    raise NotImplementedError('TODO: implement matching_as_permutation (see the reference in src/mlbook)')
