"""Sensor fusion at three levels: late (boxes), early (point painting), intermediate (attention).

* **Late fusion** operates on each sensor's detections.  Weighted Boxes Fusion (Solovyev et
  al., 2021) clusters boxes by IoU and averages coordinates weighted by confidence;
  probabilistic fusion combines per-sensor detection probabilities in log-odds space.
* **Early fusion** — PointPainting (Vora et al., CVPR 2020): project each LiDAR point into
  the image and append the semantic scores of the pixel it hits, so a point-cloud detector
  sees ``(x, y, z, p_1..p_K)``.
* **Intermediate fusion** — TransFusion / BEVFusion-style cross-attention: LiDAR BEV tokens
  are queries, camera tokens are keys/values; a key-padding mask implements sensor dropout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mlbook.perception.camera_rig import Camera

# ---------------------------------------------------------------------------
# Late fusion
# ---------------------------------------------------------------------------


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of axis-aligned boxes ``(x_min, y_min, x_max, y_max)``: (N, 4), (M, 4) → (N, M)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # (N, M, 2) top-left of intersection
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2) bottom-right
    wh = np.clip(rb - lt, 0.0, None)  # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)  # (N, M)


def weighted_boxes_fusion(boxes_per_model: list[np.ndarray], scores_per_model: list[np.ndarray],
                          iou_thr: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """WBF for one class.  Returns fused boxes (F, 4) and scores (F,).

    Every box (from any model) is visited in decreasing confidence.  It joins the first
    cluster whose *current fused box* overlaps it with IoU ≥ ``iou_thr``, else starts a new
    cluster.  Fused box = confidence-weighted mean; fused score = mean confidence scaled by
    ``min(n_boxes, n_models) / n_models`` so a box seen by one of three sensors is penalised.
    """
    n_models = len(boxes_per_model)
    all_boxes = np.concatenate(boxes_per_model, axis=0)  # (T, 4)
    all_scores = np.concatenate(scores_per_model, axis=0)  # (T,)
    order = np.argsort(-all_scores)  # (T,)
    clusters: list[list[int]] = []
    fused: list[np.ndarray] = []  # each (4,)
    for idx in order:
        box, score = all_boxes[idx], all_scores[idx]
        if fused:
            ious = box_iou(box[None, :], np.stack(fused))[0]  # (F,)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                clusters[j].append(int(idx))
                members = clusters[j]
                w = all_scores[members]  # (m,)
                fused[j] = (all_boxes[members] * w[:, None]).sum(0) / w.sum()  # (4,) weighted mean
                continue
        clusters.append([int(idx)])
        fused.append(box.copy())
    out_scores = np.array([all_scores[m].mean() * min(len(m), n_models) / n_models for m in clusters])  # (F,)
    return np.stack(fused) if fused else np.zeros((0, 4)), out_scores


def fuse_detection_probabilities(probs: np.ndarray, prior: float = 0.5) -> np.ndarray:
    """Independent-evidence fusion in log-odds: ``logit(p) = Σ_i logit(p_i) − (n−1)·logit(prior)``.

    Args:
        probs: (n_sensors, N) per-sensor probabilities that object ``j`` exists.
    Returns:
        (N,) fused probabilities.  Two sensors at 0.7 with prior 0.5 give ≈ 0.845, more than
        either alone — the point of redundancy; sensors that report 0.5 contribute nothing.
    """
    eps = 1e-6
    p = np.clip(probs, eps, 1.0 - eps)  # (n, N)
    logit = np.log(p / (1.0 - p))  # (n, N)
    prior_logit = np.log(prior / (1.0 - prior))
    fused_logit = logit.sum(0) - (p.shape[0] - 1) * prior_logit  # (N,)
    return 1.0 / (1.0 + np.exp(-fused_logit))  # (N,)


# ---------------------------------------------------------------------------
# Early fusion: PointPainting
# ---------------------------------------------------------------------------


def point_painting(points_ego: np.ndarray, seg_scores: np.ndarray, camera: Camera) -> np.ndarray:
    """Append per-pixel semantic scores to every LiDAR point that projects into the image.

    Args:
        points_ego: (N, 3).  seg_scores: (K, H, W) class probabilities of the image.
    Returns:
        (N, 3 + K).  Points outside the image (or behind the camera) get zeros — with a
        multi-camera rig you call this once per camera and take the hit that has a valid
        projection (a point rarely lands in two images).
    """
    k, h, w = seg_scores.shape
    pixels, _, valid = camera.project(points_ego)  # (N, 2), (N,), (N,)
    painted = np.zeros((points_ego.shape[0], k))  # (N, K)
    u = np.clip(pixels[valid, 0].astype(np.int64), 0, w - 1)  # (M,) nearest pixel column
    v = np.clip(pixels[valid, 1].astype(np.int64), 0, h - 1)  # (M,)
    painted[valid] = seg_scores[:, v, u].T  # (M, K)
    return np.concatenate([points_ego, painted], axis=1)  # (N, 3 + K)


def radar_points_to_bev(points: np.ndarray, bev_extent: tuple[float, float, float, float],
                        bev_hw: tuple[int, int]) -> np.ndarray:
    """Rasterise radar returns ``(x, y, v_radial, rcs)`` into a 3-channel BEV image.

    Channels: [0] hit count, [1] mean radial velocity, [2] max RCS.  Shape (3, X, Y).
    Radar gives ~100s of points per sweep (vs ~100k LiDAR), so most cells stay empty —
    which is why radar is usually fused as a BEV image rather than as a point set.
    """
    x_min, x_max, y_min, y_max = bev_extent
    nx, ny = bev_hw
    bev = np.zeros((3, nx, ny))  # (3, X, Y)
    ix = np.floor((points[:, 0] - x_min) / (x_max - x_min) * nx).astype(np.int64)  # (N,)
    iy = np.floor((points[:, 1] - y_min) / (y_max - y_min) * ny).astype(np.int64)  # (N,)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)  # (N,)
    for i, j, vr, rcs in zip(ix[inside], iy[inside], points[inside, 2], points[inside, 3]):
        n = bev[0, i, j]
        bev[1, i, j] = (bev[1, i, j] * n + vr) / (n + 1)  # running mean of v_radial
        bev[2, i, j] = max(bev[2, i, j], rcs) if n > 0 else rcs
        bev[0, i, j] = n + 1
    return bev


# ---------------------------------------------------------------------------
# Intermediate fusion: cross-attention (TransFusion / BEVFusion flavour)
# ---------------------------------------------------------------------------


class CrossAttentionFusion(nn.Module):
    """LiDAR tokens attend to camera tokens; multi-head, explicit Q/K/V, key-padding mask.

    forward(lidar (B, Nq, d), camera (B, Nk, d), camera_mask (B, Nk) bool = key is *missing*)
    → (B, Nq, d).  When every camera key is masked (camera failure) the output is the LiDAR
    input unchanged, so the block degrades to LiDAR-only instead of producing garbage.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, lidar: torch.Tensor, camera: torch.Tensor, camera_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, nq, d = lidar.shape
        nk = camera.shape[1]
        q = self.q_proj(lidar).view(b, nq, self.h, self.d_head).transpose(1, 2)  # (B, H, Nq, d_head)
        k = self.k_proj(camera).view(b, nk, self.h, self.d_head).transpose(1, 2)  # (B, H, Nk, d_head)
        v = self.v_proj(camera).view(b, nk, self.h, self.d_head).transpose(1, 2)  # (B, H, Nk, d_head)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head ** 0.5)  # (B, H, Nq, Nk)
        if camera_mask is not None:
            scores = scores.masked_fill(camera_mask.view(b, 1, 1, nk), float("-inf"))  # (B, H, Nq, Nk)
        attn = torch.softmax(scores, dim=-1)  # (B, H, Nq, Nk)
        attn = torch.nan_to_num(attn, nan=0.0)  # all keys masked → zero context, not NaN
        ctx = torch.matmul(attn, v)  # (B, H, Nq, d_head)
        ctx = ctx.transpose(1, 2).reshape(b, nq, d)  # (B, Nq, d) concat heads
        return lidar + self.out_proj(ctx)  # (B, Nq, d) residual
