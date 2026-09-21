"""Feed-forward multi-view geometry: pointmaps instead of a correspondence pipeline (PyTorch).

Classical SfM answers "where is this scene" by chaining detect, match, verify, triangulate
and optimise.  The feed-forward family (DUSt3R, MASt3R, VGGT and relatives) replaces the
chain with a single regression: two images in, and for each one a **pointmap**, a dense
``(3, H, W)`` grid giving the 3-D location of every pixel *expressed in the first camera's
frame*.

That output choice is what makes the approach work.  Because both pointmaps live in one
frame, the network has implicitly solved correspondence, relative pose and depth at once,
and each of those can be read back out with closed-form algebra rather than another network:

* **depth** is the z channel of a camera's own pointmap,
* **relative pose** is the similarity aligning one camera's pointmap in its own frame with
  the same points expressed in the other's (weighted Umeyama, one SVD),
* **focal length** has a closed form given the pointmap, assuming a centred principal point.

The catch: a monocular-scale ambiguity remains, so training uses a scale-normalised loss
and the prediction is metric only after external scale comes in.  For a delivery decision
that means GPS baselines, a laser altimeter or ground control, never the network alone.

Shapes: images ``(B, 3, H, W)``, pointmaps ``(B, 3, H, W)``, confidence ``(B, 1, H, W)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ the loss and its parts


def normalise_scale(points: torch.Tensor, valid: torch.Tensor | None = None,
                    eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Divide a pointmap by its own mean distance from the origin.

    $$z = \frac{1}{|V|}\sum_{i \in V} \|X_i\|, \qquad \hat{X} = X / z.$$

    A monocular pair cannot observe absolute scale, so a loss on raw coordinates punishes
    the network for an ambiguity it cannot resolve.  Normalising both prediction and target
    the same way makes the loss scale invariant and leaves scale to be supplied later.

    Args:
        points: (B, 3, H, W).  valid: (B, 1, H, W) bool or float mask, optional.
    Returns:
        normalised pointmap (B, 3, H, W) and the per-sample scale (B, 1, 1, 1).
    """
    dist = points.norm(dim=1, keepdim=True)                        # (B, 1, H, W)
    if valid is None:
        scale = dist.mean(dim=(1, 2, 3), keepdim=True)             # (B, 1, 1, 1)
    else:
        w = valid.to(points.dtype)                                 # (B, 1, H, W)
        scale = (dist * w).sum(dim=(1, 2, 3), keepdim=True) / w.sum(dim=(1, 2, 3), keepdim=True).clamp(min=eps)
    scale = scale.clamp(min=eps)
    return points / scale, scale


def confidence_weighted_loss(
    pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor,
    alpha: float = 0.2, valid: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""DUSt3R-style confidence loss over scale-normalised pointmaps.

    $$\mathcal{L} = \frac{1}{|V|}\sum_i \Big( c_i \,\|\hat{X}_i - \hat{X}^{gt}_i\| - \alpha \log c_i \Big).$$

    The network is allowed to say "I do not know here" by lowering $c_i$, and the
    $-\alpha \log c_i$ term is the price of saying it.  Without that term the optimum is
    $c \to 0$ everywhere; without the confidence the sky, water and moving vegetation
    dominate the gradient.  Predicting *where the geometry is trustworthy* is half the
    value of these models in a safety pipeline.

    Args:
        pred, target: (B, 3, H, W) in the same frame.  conf: (B, 1, H, W), strictly positive.
        valid: (B, 1, H, W) optional mask of pixels with ground truth.
    Returns:
        scalar loss.
    """
    pred_n, _ = normalise_scale(pred, valid)                       # (B, 3, H, W)
    target_n, _ = normalise_scale(target, valid)                   # (B, 3, H, W)
    err = (pred_n - target_n).norm(dim=1, keepdim=True)            # (B, 1, H, W) euclidean
    per_pixel = conf * err - alpha * torch.log(conf)               # (B, 1, H, W)
    if valid is None:
        return per_pixel.mean()
    w = valid.to(pred.dtype)                                       # (B, 1, H, W)
    return (per_pixel * w).sum() / w.sum().clamp(min=1.0)


# --------------------------------------------------------- reading geometry back out


def pointmap_to_depth(points: torch.Tensor) -> torch.Tensor:
    """Depth map = the z channel of a camera's pointmap in its own frame. (B,3,H,W)->(B,1,H,W)."""
    return points[:, 2:3]                                          # (B, 1, H, W)


def estimate_focal(points: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    r"""Closed-form focal length from a pointmap, assuming a centred principal point.

    Every pixel $(u, v)$ should satisfy $u - c_x = f\,x/z$ and $v - c_y = f\,y/z$.  One
    unknown, many equations, weighted least squares:

    $$f^\star = \frac{\sum_i c_i \big[(u_i - c_x)\tfrac{x_i}{z_i} + (v_i - c_y)\tfrac{y_i}{z_i}\big]}
                     {\sum_i c_i \big[(\tfrac{x_i}{z_i})^2 + (\tfrac{y_i}{z_i})^2\big]}.$$

    Args:
        points: (B, 3, H, W) pointmap in the camera's own frame.
        conf: (B, 1, H, W) optional weights.
    Returns:
        (B,) focal length in pixels.
    """
    B, _, H, W = points.shape
    v, u = torch.meshgrid(torch.arange(H, dtype=points.dtype, device=points.device) + 0.5,
                          torch.arange(W, dtype=points.dtype, device=points.device) + 0.5,
                          indexing="ij")                           # (H, W) each
    du = (u - W / 2.0)[None, None]                                 # (1, 1, H, W)
    dv = (v - H / 2.0)[None, None]                                 # (1, 1, H, W)
    z = points[:, 2:3].clamp(min=1e-6)                             # (B, 1, H, W)
    xn, yn = points[:, 0:1] / z, points[:, 1:2] / z                # (B, 1, H, W) each
    w = torch.ones_like(z) if conf is None else conf               # (B, 1, H, W)
    num = (w * (du * xn + dv * yn)).sum(dim=(1, 2, 3))             # (B,)
    den = (w * (xn**2 + yn**2)).sum(dim=(1, 2, 3)).clamp(min=1e-12)  # (B,)
    return num / den                                               # (B,)


def weighted_umeyama(
    src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor | None = None,
    with_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Weighted similarity ``dst ≈ s R src + t`` in closed form, batched and differentiable.

    The confidence channel is exactly the weight to use here: pixels the network distrusts
    should not steer the pose.

    Args:
        src, dst: (B, N, 3).  weights: (B, N) non-negative, optional.
    Returns:
        R (B, 3, 3), t (B, 3), s (B,).
    """
    B, N, _ = src.shape
    w = torch.ones(B, N, dtype=src.dtype, device=src.device) if weights is None else weights
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)            # (B, N) normalised
    mu_s = (w[..., None] * src).sum(dim=1)                          # (B, 3)
    mu_d = (w[..., None] * dst).sum(dim=1)                          # (B, 3)
    Xs, Xd = src - mu_s[:, None], dst - mu_d[:, None]               # (B, N, 3) centred
    # H[b] = sum_n w[b,n] * outer(Xd[b,n], Xs[b,n]): the weighted cross-covariance
    H = torch.einsum("bn,bni,bnj->bij", w, Xd, Xs)                  # (B, 3, 3)
    U, D, Vh = torch.linalg.svd(H)
    det = torch.det(U @ Vh)                                         # (B,) +1 or -1
    S = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = U @ S @ Vh                                                  # (B, 3, 3) forced into SO(3)
    var_s = (w * (Xs**2).sum(dim=-1)).sum(dim=1).clamp(min=1e-12)   # (B,)
    s = ((D * torch.stack([torch.ones_like(det), torch.ones_like(det), det], -1)).sum(-1) / var_s
         if with_scale else torch.ones_like(var_s))                 # (B,)
    t = mu_d - s[:, None] * torch.einsum("bij,bj->bi", R, mu_s)     # (B, 3)
    return R, t, s


def relative_pose_from_pointmaps(
    self_points: torch.Tensor, other_points: torch.Tensor, conf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Relative pose from the two pointmaps of the same camera, no matching required.

    ``self_points`` is camera 2's pointmap in camera 2's own frame; ``other_points`` is the
    *same pixels* predicted in camera 1's frame.  The similarity taking one to the other is
    the relative pose, up to the residual scale ambiguity.

    Args:
        self_points, other_points: (B, 3, H, W).  conf: (B, 1, H, W) optional.
    Returns:
        R (B, 3, 3), t (B, 3), s (B,) with ``X_cam1 ≈ s R X_cam2 + t``.
    """
    B = self_points.shape[0]
    src = self_points.reshape(B, 3, -1).transpose(1, 2)            # (B, N, 3)
    dst = other_points.reshape(B, 3, -1).transpose(1, 2)           # (B, N, 3)
    w = None if conf is None else conf.reshape(B, -1)              # (B, N)
    return weighted_umeyama(src, dst, w)


# --------------------------------------------------------------------- the tiny network


class CrossViewAttention(nn.Module):
    """One cross-attention block: view A's tokens query view B's tokens.

    This is the only place the two views talk.  Stacking several of these, with self- and
    cross-attention interleaved, is the whole architectural idea behind the feed-forward
    reconstruction models; the rest is a ViT encoder and a dense head.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)                               # separate projections,
        self.k = nn.Linear(dim, dim)                               # not one fused 3*dim
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.scale = dim**-0.5

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: (B, N, C) tokens of the two views. Returns (B, N, C)."""
        qa = self.q(self.norm_a(a))                                # (B, N, C)
        kb = self.k(self.norm_b(b))                                # (B, N, C)
        vb = self.v(self.norm_b(b))                                # (B, N, C)
        attn = torch.softmax(qa @ kb.transpose(1, 2) * self.scale, dim=-1)  # (B, N, N)
        return a + self.out(attn @ vb)                             # (B, N, C) residual


class PointmapNet(nn.Module):
    """A deliberately small two-view pointmap regressor with the right output structure.

    Both heads predict in **view 1's frame**, which is what removes the need for an explicit
    pose stage.  Real models use a ViT encoder, many attention blocks and a DPT head; the
    shapes, the frame convention and the confidence channel are identical.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),       # (B, w, H/2, W/2)
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),   # (B, w, H/4, W/4)
        )
        self.cross = CrossViewAttention(width)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.ReLU(),  # (B, w, H/2, W/2)
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.ReLU(),  # (B, w, H, W)
            nn.Conv2d(width, 4, 1),                                                # (B, 4, H, W)
        )

    def _tokens(self, feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        B, C, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2), (h, w)             # (B, h*w, C), (h, w)

    def _head(self, tokens: torch.Tensor, hw: tuple[int, int], B: int, C: int):
        grid = tokens.transpose(1, 2).reshape(B, C, hw[0], hw[1])  # (B, C, h, w)
        out = self.decode(grid)                                    # (B, 4, H, W)
        points = out[:, :3]                                        # (B, 3, H, W)
        conf = 1.0 + F.softplus(out[:, 3:4])                       # (B, 1, H, W) >= 1
        return points, conf

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """img1, img2: (B, 3, H, W). Returns two (pointmap, conf) pairs, both in view 1's frame."""
        f1, f2 = self.encode(img1), self.encode(img2)              # (B, w, H/4, W/4) each
        t1, hw = self._tokens(f1)                                  # (B, N, w)
        t2, _ = self._tokens(f2)                                   # (B, N, w)
        B, C = t1.shape[0], t1.shape[2]
        a1 = self.cross(t1, t2)                                    # (B, N, w) view 1 sees view 2
        a2 = self.cross(t2, t1)                                    # (B, N, w) and vice versa
        return self._head(a1, hw, B, C), self._head(a2, hw, B, C)
