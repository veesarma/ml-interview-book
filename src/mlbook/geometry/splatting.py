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
    q = q / q.norm(dim=-1, keepdim=True)                       # (N, 4) normalise
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]            # (N,) each
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)                                                  # (N, 9)
    return R.reshape(-1, 3, 3)                                  # (N, 3, 3)


def covariance_from_scale_rotation(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    r"""Build $\Sigma = R S S^\top R^\top$ from per-axis scales and an orientation.

    Parametrising the covariance this way rather than storing six free numbers is what
    keeps it positive semi-definite under gradient descent: any ``scales`` and any ``quats``
    give a valid covariance, so the optimiser can never step outside the feasible set.

    Args:
        scales: (N, 3) positive per-axis standard deviations.  quats: (N, 4).
    Returns:
        (N, 3, 3) covariances.
    """
    R = quaternion_to_rotation(quats)                           # (N, 3, 3)
    S = torch.diag_embed(scales)                                # (N, 3, 3)
    M = R @ S                                                   # (N, 3, 3)
    return M @ M.transpose(1, 2)                                # (N, 3, 3) = R S S^T R^T


def project_means(mu: torch.Tensor, K: torch.Tensor, R: torch.Tensor,
                  t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pinhole-project Gaussian centres.

    Args:
        mu: (N, 3) world centres.  K: (3, 3).  R: (3, 3), t: (3,) with ``P_c = R P_w + t``.
    Returns:
        uv (N, 2) pixels, depth (N,) camera-frame z, mu_cam (N, 3).
    """
    mu_cam = mu @ R.T + t                                       # (N, 3) camera frame
    uv_h = mu_cam @ K.T                                         # (N, 3) homogeneous pixels
    uv = uv_h[:, :2] / uv_h[:, 2:3].clamp(min=1e-6)             # (N, 2)
    return uv, mu_cam[:, 2], mu_cam                             # (N, 2), (N,), (N, 3)


def project_covariance(mu_cam: torch.Tensor, cov: torch.Tensor, K: torch.Tensor,
                       R: torch.Tensor, dilation: float = 0.3) -> torch.Tensor:
    r"""Linearised (EWA) projection of a 3-D covariance into image space.

    $$J = \begin{bmatrix} f_x/z & 0 & -f_x x / z^2 \\ 0 & f_y/z & -f_y y / z^2 \end{bmatrix},
      \qquad \Sigma_{2D} = J W \Sigma W^\top J^\top + \epsilon I.$$

    The ``dilation`` term is the low-pass filter every splat rasteriser adds: without it a
    Gaussian thinner than a pixel aliases into flicker as the camera moves.

    Args:
        mu_cam: (N, 3) centres in camera coordinates.  cov: (N, 3, 3) world covariances.
    Returns:
        (N, 2, 2) image-space covariances, in pixels squared.
    """
    fx, fy = K[0, 0], K[1, 1]
    x, y, z = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2].clamp(min=1e-6)   # (N,) each
    J = torch.zeros(len(mu_cam), 2, 3, dtype=mu_cam.dtype, device=mu_cam.device)
    J[:, 0, 0], J[:, 1, 1] = fx / z, fy / z
    J[:, 0, 2], J[:, 1, 2] = -fx * x / z**2, -fy * y / z**2              # (N, 2, 3)
    W = J @ R                                                            # (N, 2, 3) = J W
    cov2d = W @ cov @ W.transpose(1, 2)                                  # (N, 2, 2)
    eye = torch.eye(2, dtype=cov2d.dtype, device=cov2d.device)
    return cov2d + dilation * eye                                        # (N, 2, 2)


def gaussian_weights(uv: torch.Tensor, cov2d: torch.Tensor, height: int,
                     width: int) -> torch.Tensor:
    r"""Evaluate every projected Gaussian at every pixel centre.

    $$G_i(p) = \exp\!\left(-\tfrac{1}{2}(p - \mu_i)^\top \Sigma_{2D,i}^{-1} (p - \mu_i)\right)$$

    with no normalising constant: opacity carries the amplitude, so a splat that shrinks
    does not automatically get brighter.

    Returns:
        (N, H, W) weights in [0, 1].
    """
    ys = torch.arange(height, dtype=uv.dtype, device=uv.device) + 0.5     # (H,) pixel centres
    xs = torch.arange(width, dtype=uv.dtype, device=uv.device) + 0.5      # (W,)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")                # (H, W) each
    d = torch.stack([grid_x, grid_y], dim=-1)[None] - uv[:, None, None, :]  # (N, H, W, 2)
    inv = torch.linalg.inv(cov2d)                                          # (N, 2, 2)
    # mahalanobis = d^T inv d, written out so the quadratic form stays readable
    m = (d[..., 0] ** 2 * inv[:, None, None, 0, 0]
         + 2 * d[..., 0] * d[..., 1] * inv[:, None, None, 0, 1]
         + d[..., 1] ** 2 * inv[:, None, None, 1, 1])                      # (N, H, W)
    return torch.exp(-0.5 * m)                                             # (N, H, W)


def alpha_composite(colors: torch.Tensor, alphas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Front-to-back "over" compositing of depth-sorted samples.

    $$C = \sum_i c_i \alpha_i \prod_{j<i} (1 - \alpha_j), \qquad
      T_{\text{final}} = \prod_i (1 - \alpha_i).$$

    The same operator appears in NeRF's quadrature; splatting differs in where the samples
    come from, not in how they are combined.

    Args:
        colors: (N, C, ...) sample colours, already sorted front to back.
        alphas: (N, ...) per-sample opacity in [0, 1].
    Returns:
        image (C, ...) and remaining transmittance (...,).
    """
    transmittance = torch.ones_like(alphas[0])                   # (...,) starts fully open
    image = torch.zeros_like(colors[0])                          # (C, ...)
    for i in range(len(alphas)):
        weight = alphas[i] * transmittance                       # (...,) contribution of i
        image = image + colors[i] * weight[None]
        transmittance = transmittance * (1.0 - alphas[i])
    return image, transmittance


def render_gaussians(
    mu: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, colors: torch.Tensor,
    opacities: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
    height: int, width: int, background: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable brute-force splat renderer.

    Args:
        mu: (N, 3) centres.  scales: (N, 3).  quats: (N, 4).  colors: (N, 3) in [0, 1].
        opacities: (N,) in [0, 1].  K, R, t: camera.  height, width: image size.
    Returns:
        image (3, H, W), alpha (H, W) coverage, and depth (H, W) as the alpha-weighted
        expected depth, which is the quantity to be suspicious of for safety decisions.
    """
    uv, depth, mu_cam = project_means(mu, K, R, t)               # (N, 2), (N,), (N, 3)
    cov = covariance_from_scale_rotation(scales, quats)          # (N, 3, 3)
    cov2d = project_covariance(mu_cam, cov, K, R)                # (N, 2, 2)
    order = torch.argsort(depth)                                 # (N,) front to back
    G = gaussian_weights(uv[order], cov2d[order], height, width)  # (N, H, W)
    a = (opacities[order][:, None, None] * G).clamp(max=0.999)   # (N, H, W)
    c = colors[order][:, :, None, None].expand(-1, -1, height, width)  # (N, 3, H, W)
    image, transmittance = alpha_composite(c, a)                 # (3, H, W), (H, W)
    d = depth[order][:, None, None].expand(-1, height, width)    # (N, H, W)
    depth_map, _ = alpha_composite(d[:, None], a)                # (1, H, W)
    image = image + background * transmittance[None]
    coverage = 1.0 - transmittance                               # (H, W)
    return image, coverage, depth_map[0] / coverage.clamp(min=1e-6)
