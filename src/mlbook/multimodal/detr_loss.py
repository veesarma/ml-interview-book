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
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)  # (..., 4)


def pairwise_giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalised IoU between every pair: ``a`` (N, 4), ``b`` (M, 4), xyxy -> (N, M).

    GIoU = IoU - |C \\ (A ∪ B)| / |C|, C = smallest enclosing box. Range (-1, 1].
    """
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)
    lt = torch.max(a[:, None, :2], b[None, :, :2])  # (N, M, 2) top-left of intersection
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2) bottom-right of intersection
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)
    union = area_a[:, None] + area_b[None, :] - inter  # (N, M)
    iou = inter / union  # (N, M)
    lt_c = torch.min(a[:, None, :2], b[None, :, :2])  # (N, M, 2) enclosing box
    rb_c = torch.max(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2)
    wh_c = (rb_c - lt_c).clamp(min=0)  # (N, M, 2)
    area_c = wh_c[..., 0] * wh_c[..., 1]  # (N, M)
    return iou - (area_c - union) / area_c  # (N, M)


@torch.no_grad()
def hungarian_match(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    targets: list[dict],
    w_class: float = 1.0,
    w_l1: float = 5.0,
    w_giou: float = 2.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per image, the (pred_idx, tgt_idx) pairs minimising the matching cost.

    ``logits`` (B, Q, K+1), ``boxes`` (B, Q, 4) cxcywh. Returns a list of B tuples of long tensors (T_b,).
    """
    out = []
    for b, tgt in enumerate(targets):
        T = tgt["labels"].shape[0]
        if T == 0:
            out.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
            continue
        prob = logits[b].softmax(-1)  # (Q, K+1)
        cost_class = -prob[:, tgt["labels"]]  # (Q, T): minus prob of the target's class
        cost_l1 = torch.cdist(boxes[b], tgt["boxes"], p=1)  # (Q, T)
        cost_giou = -pairwise_giou(box_cxcywh_to_xyxy(boxes[b]), box_cxcywh_to_xyxy(tgt["boxes"]))  # (Q, T)
        C = w_class * cost_class + w_l1 * cost_l1 + w_giou * cost_giou  # (Q, T)
        rows, cols = hungarian(C.cpu().numpy())
        out.append((torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)))
    return out


def detr_loss(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    targets: list[dict],
    w_l1: float = 5.0,
    w_giou: float = 2.0,
    eos_coef: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Set loss L = CE + w_l1 * L1 + w_giou * (1 - GIoU), box terms normalised by number of targets.

    ``logits`` (B, Q, K+1), ``boxes`` (B, Q, 4). Unmatched queries are trained to
    predict "no object" (class index K) with weight ``eos_coef`` in the CE.
    Returns dict with ``loss``, ``loss_ce``, ``loss_l1``, ``loss_giou``.
    """
    B, Q, K1 = logits.shape
    no_obj = K1 - 1
    match = hungarian_match(logits, boxes, targets, w_l1=w_l1, w_giou=w_giou)
    target_classes = torch.full((B, Q), no_obj, dtype=torch.long)  # (B, Q), default = no object
    matched_pred, matched_tgt = [], []
    for b, (rows, cols) in enumerate(match):
        target_classes[b, rows] = targets[b]["labels"][cols]
        matched_pred.append(boxes[b, rows])  # (T_b, 4)
        matched_tgt.append(targets[b]["boxes"][cols])  # (T_b, 4)
    class_weight = torch.ones(K1)  # (K+1,)
    class_weight[no_obj] = eos_coef
    loss_ce = torch.nn.functional.cross_entropy(logits.reshape(B * Q, K1), target_classes.reshape(B * Q), weight=class_weight)
    pred_b = torch.cat(matched_pred, dim=0)  # (T_total, 4)
    tgt_b = torch.cat(matched_tgt, dim=0)  # (T_total, 4)
    n_tgt = max(pred_b.shape[0], 1)
    loss_l1 = (pred_b - tgt_b).abs().sum() / n_tgt
    giou = pairwise_giou(box_cxcywh_to_xyxy(pred_b), box_cxcywh_to_xyxy(tgt_b)).diagonal()  # (T_total,)
    loss_giou = (1.0 - giou).sum() / n_tgt
    total = loss_ce + w_l1 * loss_l1 + w_giou * loss_giou
    return {"loss": total, "loss_ce": loss_ce, "loss_l1": loss_l1, "loss_giou": loss_giou}


def matching_as_permutation(match: list[tuple[torch.Tensor, torch.Tensor]], Q: int) -> list[np.ndarray]:
    """For tests: turn each image's (rows, cols) into an array ``perm`` (Q,) with perm[q] = target index or -1."""
    out = []
    for rows, cols in match:
        perm = -np.ones(Q, dtype=int)
        perm[rows.numpy()] = cols.numpy()
        out.append(perm)
    return out
