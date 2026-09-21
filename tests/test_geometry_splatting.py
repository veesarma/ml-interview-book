"""Gaussian splatting: quaternion/covariance parametrisation, the EWA projection,
front-to-back compositing, occlusion ordering, and that gradients reach every parameter."""

import numpy as np
import torch

from mlbook.geometry import splatting as SP


def _camera(dtype=torch.float64):
    K = torch.tensor([[200.0, 0.0, 32.0], [0.0, 200.0, 32.0], [0.0, 0.0, 1.0]], dtype=dtype)
    R = torch.eye(3, dtype=dtype)                       # camera frame == world frame
    t = torch.zeros(3, dtype=dtype)
    return K, R, t


def test_quaternion_to_rotation_is_orthonormal_and_matches_a_known_case():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0],                      # identity
                      [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]],  # 90 deg about z
                     dtype=torch.float64)
    R = SP.quaternion_to_rotation(q)                              # (2, 3, 3)
    assert torch.allclose(R[0], torch.eye(3, dtype=torch.float64), atol=1e-12)
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                            dtype=torch.float64)
    assert torch.allclose(R[1], expected, atol=1e-12)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(2, 3, 3),
                          atol=1e-12)


def test_covariance_is_psd_and_isotropic_scales_give_sigma_squared_identity():
    scales = torch.tensor([[0.5, 0.5, 0.5], [2.0, 0.3, 1.0]], dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.2, 0.4, -0.3, 0.8]], dtype=torch.float64)
    cov = SP.covariance_from_scale_rotation(scales, quats)        # (2, 3, 3)

    assert torch.allclose(cov[0], 0.25 * torch.eye(3, dtype=torch.float64), atol=1e-12)
    for c in cov:                                                  # symmetric and PSD
        assert torch.allclose(c, c.T, atol=1e-12)
        assert torch.linalg.eigvalsh(c).min() > 0
    # the anisotropic one keeps its volume under rotation: det = prod(s^2)
    assert torch.isclose(torch.det(cov[1]), torch.tensor((2.0 * 0.3 * 1.0) ** 2,
                                                         dtype=torch.float64))


def test_ewa_projection_shrinks_with_depth_like_one_over_z_squared():
    K, R, t = _camera()
    sigma = 0.1
    mu = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 10.0]], dtype=torch.float64)
    scales = torch.full((2, 3), sigma, dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
    cov = SP.covariance_from_scale_rotation(scales, quats)
    _, _, mu_cam = SP.project_means(mu, K, R, t)
    cov2d = SP.project_covariance(mu_cam, cov, K, R, dilation=0.0)   # (2, 2, 2)

    # on axis the Jacobian is diag(f/z, f/z), so Sigma_2D = (f sigma / z)^2 I
    for i, z in enumerate([5.0, 10.0]):
        expected = (200.0 * sigma / z) ** 2
        assert torch.allclose(cov2d[i], expected * torch.eye(2, dtype=torch.float64), atol=1e-9)
    # doubling the depth quarters the image-space area
    assert torch.isclose(torch.det(cov2d[0]) / torch.det(cov2d[1]),
                         torch.tensor(16.0, dtype=torch.float64))


def test_gaussian_weight_peaks_at_the_projected_centre():
    K, R, t = _camera()
    mu = torch.tensor([[0.3, -0.2, 4.0]], dtype=torch.float64)
    scales = torch.full((1, 3), 0.08, dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    uv, _, mu_cam = SP.project_means(mu, K, R, t)
    cov2d = SP.project_covariance(mu_cam, SP.covariance_from_scale_rotation(scales, quats), K, R)
    G = SP.gaussian_weights(uv, cov2d, 64, 64)                     # (1, 64, 64)

    row, col = np.unravel_index(int(torch.argmax(G)), (64, 64))
    # pixel centres are at +0.5, so the peak pixel's centre is within half a pixel of uv
    assert abs((col + 0.5) - float(uv[0, 0])) <= 0.5
    assert abs((row + 0.5) - float(uv[0, 1])) <= 0.5
    assert 0.0 < G.max() <= 1.0


def test_alpha_composite_matches_the_over_operator_by_hand():
    colors = torch.tensor([[[1.0]], [[0.0]], [[0.5]]], dtype=torch.float64)  # (3, 1, 1)
    alphas = torch.tensor([[0.5], [0.5], [1.0]], dtype=torch.float64)        # (3, 1)
    image, T = SP.alpha_composite(colors, alphas)
    # 1*0.5 + 0*0.5*0.5 + 0.5*1.0*0.25 = 0.625, and T = 0.5*0.5*0 = 0
    assert torch.allclose(image, torch.tensor([[0.625]], dtype=torch.float64))
    assert torch.allclose(T, torch.zeros(1, dtype=torch.float64))


def test_an_opaque_near_splat_occludes_the_one_behind_it():
    K, R, t = _camera()
    mu = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]], dtype=torch.float64)
    scales = torch.full((2, 3), 0.15, dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)  # red near
    opac = torch.tensor([0.999, 0.999], dtype=torch.float64)

    img, cov, depth = SP.render_gaussians(mu, scales, quats, colors, opac, K, R, t, 64, 64)
    centre = img[:, 32, 32]                                         # (3,)

    assert centre[0] > 0.99 and centre[2] < 0.01                    # red wins at the centre
    assert cov[32, 32] > 0.99
    assert abs(float(depth[32, 32]) - 2.0) < 0.05                   # depth is the near surface

    # swap the depths and the blue splat is now in front
    mu_swapped = mu.flip(0)
    img2, _, _ = SP.render_gaussians(mu_swapped, scales, quats, colors, opac, K, R, t, 64, 64)
    assert img2[2, 32, 32] > 0.99 and img2[0, 32, 32] < 0.01


def test_render_is_differentiable_in_every_parameter():
    K, R, t = _camera(dtype=torch.float32)
    mu = torch.tensor([[0.1, 0.0, 3.0], [-0.2, 0.1, 5.0]], requires_grad=True)
    # anisotropic on purpose: R S S^T R^T is rotation-invariant when the scales are
    # equal, so an isotropic splat has an exactly zero gradient on its quaternion
    scales = torch.tensor([[0.2, 0.08, 0.12], [0.05, 0.18, 0.1]], requires_grad=True)
    quats = torch.tensor([[0.92, 0.1, -0.2, 0.3], [1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    colors = torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.3, 0.9]], requires_grad=True)
    opac = torch.tensor([0.6, 0.7], requires_grad=True)

    img, _, _ = SP.render_gaussians(mu, scales, quats, colors, opac, K, R, t, 32, 32)
    img.sum().backward()

    for name, p in [("mu", mu), ("scales", scales), ("quats", quats),
                    ("colors", colors), ("opacities", opac)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def _fit_mu(start, target_mu, steps=150, lr=0.02):
    """Fit one splat's centre to a rendered target and return (losses, final mu)."""
    K, R, t = _camera(dtype=torch.float32)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    scales = torch.full((1, 3), 0.1)
    colors = torch.ones(1, 3)
    opac = torch.tensor([0.95])
    target, _, _ = SP.render_gaussians(target_mu, scales, quats, colors, opac, K, R, t, 48, 48)

    mu = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        img, _, _ = SP.render_gaussians(mu, scales, quats, colors, opac, K, R, t, 48, 48)
        loss = ((img - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    return losses, mu.detach()


def test_photometric_fit_converges_when_the_splat_already_overlaps_its_target():
    torch.manual_seed(0)
    target_mu = torch.tensor([[0.25, -0.15, 4.0]])
    start = torch.tensor([[0.18, -0.08, 4.0]])          # ~5 px away, inside the support
    losses, mu = _fit_mu(start, target_mu)

    assert losses[-1] < 0.02 * losses[0]
    assert torch.allclose(mu[0, :2], target_mu[0, :2], atol=0.02)


def test_photometric_fit_stalls_when_the_splat_misses_its_target_entirely():
    """The local-support failure that densification and coarse-to-fine exist to fix.

    A splat with sigma ~5 px started 25 px from the target overlaps it nowhere, so the
    photometric gradient is numerically zero and Adam has nothing to descend.  Real 3DGS
    handles this by splitting, cloning and pruning splats rather than by relying on the
    position gradient alone.
    """
    torch.manual_seed(0)
    target_mu = torch.tensor([[0.25, -0.15, 4.0]])
    start = torch.tensor([[-0.10, 0.20, 4.0]])          # ~25 px away, no overlap
    losses, mu = _fit_mu(start, target_mu)

    assert losses[-1] > 0.3 * losses[0]                  # barely moved the loss
    assert not torch.allclose(mu[0, :2], target_mu[0, :2], atol=0.05)
