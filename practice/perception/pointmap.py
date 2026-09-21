# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/pointmap.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k pointmap -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/pointmap --force

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

def normalise_scale(points: torch.Tensor, valid: torch.Tensor | None=None, eps: float=1e-06) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide a pointmap by its own mean distance from the origin.

    $$z = \\frac{1}{|V|}\\sum_{i \\in V} \\|X_i\\|, \\qquad \\hat{X} = X / z.$$

    A monocular pair cannot observe absolute scale, so a loss on raw coordinates punishes
    the network for an ambiguity it cannot resolve.  Normalising both prediction and target
    the same way makes the loss scale invariant and leaves scale to be supplied later.

    Args:
        points: (B, 3, H, W).  valid: (B, 1, H, W) bool or float mask, optional.
    Returns:
        normalised pointmap (B, 3, H, W) and the per-sample scale (B, 1, 1, 1).
    """
    raise NotImplementedError('TODO: implement normalise_scale (see the reference in src/mlbook)')

def confidence_weighted_loss(pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor, alpha: float=0.2, valid: torch.Tensor | None=None) -> torch.Tensor:
    """DUSt3R-style confidence loss over scale-normalised pointmaps.

    $$\\mathcal{L} = \\frac{1}{|V|}\\sum_i \\Big( c_i \\,\\|\\hat{X}_i - \\hat{X}^{gt}_i\\| - \\alpha \\log c_i \\Big).$$

    The network is allowed to say "I do not know here" by lowering $c_i$, and the
    $-\\alpha \\log c_i$ term is the price of saying it.  Without that term the optimum is
    $c \\to 0$ everywhere; without the confidence the sky, water and moving vegetation
    dominate the gradient.  Predicting *where the geometry is trustworthy* is half the
    value of these models in a safety pipeline.

    Args:
        pred, target: (B, 3, H, W) in the same frame.  conf: (B, 1, H, W), strictly positive.
        valid: (B, 1, H, W) optional mask of pixels with ground truth.
    Returns:
        scalar loss.
    """
    raise NotImplementedError('TODO: implement confidence_weighted_loss (see the reference in src/mlbook)')

def pointmap_to_depth(points: torch.Tensor) -> torch.Tensor:
    """Depth map = the z channel of a camera's pointmap in its own frame. (B,3,H,W)->(B,1,H,W)."""
    raise NotImplementedError('TODO: implement pointmap_to_depth (see the reference in src/mlbook)')

def estimate_focal(points: torch.Tensor, conf: torch.Tensor | None=None) -> torch.Tensor:
    """Closed-form focal length from a pointmap, assuming a centred principal point.

    Every pixel $(u, v)$ should satisfy $u - c_x = f\\,x/z$ and $v - c_y = f\\,y/z$.  One
    unknown, many equations, weighted least squares:

    $$f^\\star = \\frac{\\sum_i c_i \\big[(u_i - c_x)\\tfrac{x_i}{z_i} + (v_i - c_y)\\tfrac{y_i}{z_i}\\big]}
                     {\\sum_i c_i \\big[(\\tfrac{x_i}{z_i})^2 + (\\tfrac{y_i}{z_i})^2\\big]}.$$

    Args:
        points: (B, 3, H, W) pointmap in the camera's own frame.
        conf: (B, 1, H, W) optional weights.
    Returns:
        (B,) focal length in pixels.
    """
    raise NotImplementedError('TODO: implement estimate_focal (see the reference in src/mlbook)')

def weighted_umeyama(src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor | None=None, with_scale: bool=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted similarity ``dst ≈ s R src + t`` in closed form, batched and differentiable.

    The confidence channel is exactly the weight to use here: pixels the network distrusts
    should not steer the pose.

    Args:
        src, dst: (B, N, 3).  weights: (B, N) non-negative, optional.
    Returns:
        R (B, 3, 3), t (B, 3), s (B,).
    """
    raise NotImplementedError('TODO: implement weighted_umeyama (see the reference in src/mlbook)')

def relative_pose_from_pointmaps(self_points: torch.Tensor, other_points: torch.Tensor, conf: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Relative pose from the two pointmaps of the same camera, no matching required.

    ``self_points`` is camera 2's pointmap in camera 2's own frame; ``other_points`` is the
    *same pixels* predicted in camera 1's frame.  The similarity taking one to the other is
    the relative pose, up to the residual scale ambiguity.

    Args:
        self_points, other_points: (B, 3, H, W).  conf: (B, 1, H, W) optional.
    Returns:
        R (B, 3, 3), t (B, 3), s (B,) with ``X_cam1 ≈ s R X_cam2 + t``.
    """
    raise NotImplementedError('TODO: implement relative_pose_from_pointmaps (see the reference in src/mlbook)')

class CrossViewAttention(nn.Module):
    """One cross-attention block: view A's tokens query view B's tokens.

    This is the only place the two views talk.  Stacking several of these, with self- and
    cross-attention interleaved, is the whole architectural idea behind the feed-forward
    reconstruction models; the rest is a ViT encoder and a dense head.
    """

    def __init__(self, dim: int):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: (B, N, C) tokens of the two views. Returns (B, N, C)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class PointmapNet(nn.Module):
    """A deliberately small two-view pointmap regressor with the right output structure.

    Both heads predict in **view 1's frame**, which is what removes the need for an explicit
    pose stage.  Real models use a ViT encoder, many attention blocks and a DPT head; the
    shapes, the frame convention and the confidence channel are identical.
    """

    def __init__(self, width: int=32):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _tokens(self, feat: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        raise NotImplementedError('TODO: implement _tokens (see the reference in src/mlbook)')

    def _head(self, tokens: torch.Tensor, hw: tuple[int, int], B: int, C: int):
        raise NotImplementedError('TODO: implement _head (see the reference in src/mlbook)')

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """img1, img2: (B, 3, H, W). Returns two (pointmap, conf) pairs, both in view 1's frame."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
