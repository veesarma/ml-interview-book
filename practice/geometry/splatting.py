# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/splatting.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k splatting -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/splatting --force

"""3-D Gaussian splatting: the projection and compositing maths, in PyTorch.

A splat scene is a set of anisotropic 3-D Gaussians, each carrying

    mu (3,)      centre in world coordinates
    Sigma (3,3)  covariance, stored as a rotation q and per-axis scales s
    c (3,)       colour (view-dependent in the real thing, via spherical harmonics)
    alpha        opacity

Rendering is the part worth being able to derive:

1. **Project the centre** with the ordinary pinhole model.
2. **Project the covariance** through the *linearised* projection.  Perspective division
   is non-linear, so an exact projected Gaussian is not Gaussian; the standard move (EWA
   splatting) takes the Jacobian $J$ of the projection at the centre and sets
   $\\Sigma_{2D} = J W \\Sigma W^\\top J^\\top$ with $W$ the world-to-camera rotation.  The
   approximation degrades far from the principal point and for splats that are large in
   depth, and that is a real source of artefacts.
3. **Sort by depth and composite front to back** with the same over operator volume
   rendering uses.

This module renders every Gaussian against every pixel, which is ``O(N H W)`` and only
sane for the small scenes in the tests.  Production rasterisers bin Gaussians into 16x16
tiles and sort per tile, which is what makes the method fast.  The maths is identical.

Why this matters for an offboard aerial stack: splats are superb at *appearance* and say
very little about *metric correctness*.  A photometric loss is happy to place a floating
blob that renders correctly from every training view and is not a surface.  Geometry for
a safety decision needs an independent check.
"""
from __future__ import annotations
import torch

def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions ``(w, x, y, z)`` to rotation matrices. (N, 4) -> (N, 3, 3)."""
    raise NotImplementedError('TODO: implement quaternion_to_rotation (see the reference in src/mlbook)')

def covariance_from_scale_rotation(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Build $\\Sigma = R S S^\\top R^\\top$ from per-axis scales and an orientation.

    Parametrising the covariance this way rather than storing six free numbers is what
    keeps it positive semi-definite under gradient descent: any ``scales`` and any ``quats``
    give a valid covariance, so the optimiser can never step outside the feasible set.

    Args:
        scales: (N, 3) positive per-axis standard deviations.  quats: (N, 4).
    Returns:
        (N, 3, 3) covariances.
    """
    raise NotImplementedError('TODO: implement covariance_from_scale_rotation (see the reference in src/mlbook)')

def project_means(mu: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pinhole-project Gaussian centres.

    Args:
        mu: (N, 3) world centres.  K: (3, 3).  R: (3, 3), t: (3,) with ``P_c = R P_w + t``.
    Returns:
        uv (N, 2) pixels, depth (N,) camera-frame z, mu_cam (N, 3).
    """
    raise NotImplementedError('TODO: implement project_means (see the reference in src/mlbook)')

def project_covariance(mu_cam: torch.Tensor, cov: torch.Tensor, K: torch.Tensor, R: torch.Tensor, dilation: float=0.3) -> torch.Tensor:
    """Linearised (EWA) projection of a 3-D covariance into image space.

    $$J = \\begin{bmatrix} f_x/z & 0 & -f_x x / z^2 \\\\ 0 & f_y/z & -f_y y / z^2 \\end{bmatrix},
      \\qquad \\Sigma_{2D} = J W \\Sigma W^\\top J^\\top + \\epsilon I.$$

    The ``dilation`` term is the low-pass filter every splat rasteriser adds: without it a
    Gaussian thinner than a pixel aliases into flicker as the camera moves.

    Args:
        mu_cam: (N, 3) centres in camera coordinates.  cov: (N, 3, 3) world covariances.
    Returns:
        (N, 2, 2) image-space covariances, in pixels squared.
    """
    raise NotImplementedError('TODO: implement project_covariance (see the reference in src/mlbook)')

def gaussian_weights(uv: torch.Tensor, cov2d: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Evaluate every projected Gaussian at every pixel centre.

    $$G_i(p) = \\exp\\!\\left(-\\tfrac{1}{2}(p - \\mu_i)^\\top \\Sigma_{2D,i}^{-1} (p - \\mu_i)\\right)$$

    with no normalising constant: opacity carries the amplitude, so a splat that shrinks
    does not automatically get brighter.

    Returns:
        (N, H, W) weights in [0, 1].
    """
    raise NotImplementedError('TODO: implement gaussian_weights (see the reference in src/mlbook)')

def alpha_composite(colors: torch.Tensor, alphas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Front-to-back "over" compositing of depth-sorted samples.

    $$C = \\sum_i c_i \\alpha_i \\prod_{j<i} (1 - \\alpha_j), \\qquad
      T_{\\text{final}} = \\prod_i (1 - \\alpha_i).$$

    The same operator appears in NeRF's quadrature; splatting differs in where the samples
    come from, not in how they are combined.

    Args:
        colors: (N, C, ...) sample colours, already sorted front to back.
        alphas: (N, ...) per-sample opacity in [0, 1].
    Returns:
        image (C, ...) and remaining transmittance (...,).
    """
    raise NotImplementedError('TODO: implement alpha_composite (see the reference in src/mlbook)')

def render_gaussians(mu: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, colors: torch.Tensor, opacities: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, height: int, width: int, background: float=0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable brute-force splat renderer.

    Args:
        mu: (N, 3) centres.  scales: (N, 3).  quats: (N, 4).  colors: (N, 3) in [0, 1].
        opacities: (N,) in [0, 1].  K, R, t: camera.  height, width: image size.
    Returns:
        image (3, H, W), alpha (H, W) coverage, and depth (H, W) as the alpha-weighted
        expected depth, which is the quantity to be suspicious of for safety decisions.
    """
    raise NotImplementedError('TODO: implement render_gaussians (see the reference in src/mlbook)')
