"""Feed-forward pointmaps: scale-invariant loss, the confidence trade-off, and the
closed-form geometry (focal length, relative pose) you read back out of the prediction."""

import numpy as np
import torch

from mlbook.perception import pointmap as PM


def _synthetic_pointmap(H=16, W=20, focal=90.0, seed=0):
    """Unproject a smooth depth map with a known focal length, principal point centred."""
    g = torch.Generator().manual_seed(seed)
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64) + 0.5,
                          torch.arange(W, dtype=torch.float64) + 0.5, indexing="ij")
    z = 5.0 + 0.5 * torch.sin(u * 0.3) + 0.3 * torch.cos(v * 0.4)      # (H, W) metres
    x = (u - W / 2.0) * z / focal                                       # (H, W)
    y = (v - H / 2.0) * z / focal                                       # (H, W)
    return torch.stack([x, y, z])[None]                                 # (1, 3, H, W)


def test_normalise_scale_is_invariant_and_reports_the_factor():
    pts = _synthetic_pointmap()
    n1, s1 = PM.normalise_scale(pts)
    n2, s2 = PM.normalise_scale(pts * 7.0)

    assert torch.allclose(n1, n2, atol=1e-10)          # same shape, different metres
    assert torch.isclose(s2 / s1, torch.tensor(7.0, dtype=torch.float64))
    assert torch.isclose(n1.norm(dim=1).mean(), torch.tensor(1.0, dtype=torch.float64))


def test_normalise_scale_respects_a_validity_mask():
    pts = _synthetic_pointmap()
    valid = torch.ones(1, 1, 16, 20, dtype=torch.bool)
    valid[..., :5] = False                              # pretend the top rows have no ground truth
    corrupted = pts.clone()
    corrupted[..., :5] *= 100.0                         # garbage where it is masked out
    _, s_clean = PM.normalise_scale(pts, valid)
    _, s_corrupt = PM.normalise_scale(corrupted, valid)
    assert torch.isclose(s_clean, s_corrupt)


def test_confidence_loss_is_lowest_at_the_truth_and_scale_invariant():
    gt = _synthetic_pointmap().float()
    conf = torch.ones(1, 1, 16, 20)
    perfect = PM.confidence_weighted_loss(gt, gt, conf)
    wrong = PM.confidence_weighted_loss(gt + 0.4, gt, conf)
    assert perfect < wrong
    # predicting the right shape at the wrong scale costs nothing: that is the point
    scaled = PM.confidence_weighted_loss(gt * 3.0, gt, conf)
    assert torch.isclose(scaled, perfect, atol=1e-5)


def test_confidence_converges_to_the_analytic_optimum():
    """d/dc [c*e - alpha*log c] = 0 gives c* = alpha / e."""
    torch.manual_seed(0)
    gt = _synthetic_pointmap().float()
    pred = gt + 0.25                                    # a fixed, uniform error
    alpha = 0.2
    raw = torch.zeros(1, 1, 16, 20, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        conf = 1.0 + torch.nn.functional.softplus(raw)
        PM.confidence_weighted_loss(pred, gt, conf, alpha=alpha).backward()
        opt.step()

    pred_n, _ = PM.normalise_scale(pred)
    gt_n, _ = PM.normalise_scale(gt)
    err = (pred_n - gt_n).norm(dim=1, keepdim=True)     # (1, 1, H, W)
    target = torch.clamp(alpha / err, min=1.0)          # the parametrisation floors conf at 1
    conf = 1.0 + torch.nn.functional.softplus(raw)
    assert torch.allclose(conf, target, atol=0.05)


def test_estimate_focal_recovers_a_known_focal_length():
    for f in (60.0, 90.0, 150.0):
        pts = _synthetic_pointmap(focal=f)
        assert torch.isclose(PM.estimate_focal(pts)[0], torch.tensor(f, dtype=torch.float64),
                             rtol=1e-9)


def test_focal_estimate_sees_only_ray_directions():
    """Scaling a whole pointmap leaves x/z and y/z untouched, so the focal is unchanged.

    The same invariance that makes the loss scale-free makes this estimator scale-free:
    it constrains the *rays*, and depth scale has to be pinned down somewhere else.
    """
    pts = _synthetic_pointmap(focal=90.0)
    f_ref = PM.estimate_focal(pts)[0]
    assert torch.isclose(PM.estimate_focal(pts * 0.4)[0], f_ref, rtol=1e-12)


def test_estimate_focal_downweights_pixels_it_is_told_to_distrust():
    pts = _synthetic_pointmap(focal=90.0)
    conf = torch.ones(1, 1, 16, 20, dtype=torch.float64)
    corrupted = pts.clone()
    corrupted[:, 0, :4] += 2.0                          # lateral error: the rays now lie
    conf[:, :, :4] = 1e-6                               # but the network flagged those rows

    assert not torch.isclose(PM.estimate_focal(corrupted)[0],
                             torch.tensor(90.0, dtype=torch.float64), rtol=1e-3)
    assert torch.isclose(PM.estimate_focal(corrupted, conf)[0],
                         torch.tensor(90.0, dtype=torch.float64), rtol=1e-3)


def test_weighted_umeyama_recovers_a_known_similarity():
    torch.manual_seed(0)
    src = torch.randn(2, 40, 3, dtype=torch.float64)
    angle = torch.tensor(0.4, dtype=torch.float64)
    R = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                      [torch.sin(angle), torch.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float64).expand(2, 3, 3)
    t = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64).expand(2, 3)
    dst = 2.5 * torch.einsum("bij,bnj->bni", R, src) + t[:, None]

    R_hat, t_hat, s_hat = PM.weighted_umeyama(src, dst)

    assert torch.allclose(R_hat, R, atol=1e-9)
    assert torch.allclose(t_hat, t, atol=1e-8)
    assert torch.allclose(s_hat, torch.full((2,), 2.5, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(torch.det(R_hat), torch.ones(2, dtype=torch.float64))


def test_weights_let_umeyama_ignore_corrupted_correspondences():
    torch.manual_seed(1)
    src = torch.randn(1, 50, 3, dtype=torch.float64)
    dst = src + torch.tensor([0.5, 0.0, -1.0], dtype=torch.float64)
    dst[:, :10] += torch.randn(1, 10, 3, dtype=torch.float64) * 20.0    # gross errors
    w = torch.ones(1, 50, dtype=torch.float64)
    w[:, :10] = 0.0

    _, t_bad, _ = PM.weighted_umeyama(src, dst, with_scale=False)
    _, t_good, _ = PM.weighted_umeyama(src, dst, w, with_scale=False)
    truth = torch.tensor([[0.5, 0.0, -1.0]], dtype=torch.float64)

    assert (t_good - truth).norm() < (t_bad - truth).norm()
    assert torch.allclose(t_good, truth, atol=1e-8)


def test_relative_pose_from_pointmaps_matches_the_pose_used_to_build_them():
    pts_cam2 = _synthetic_pointmap(seed=3)                              # (1, 3, H, W)
    angle = 0.25
    R = torch.tensor([[np.cos(angle), 0.0, np.sin(angle)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(angle), 0.0, np.cos(angle)]], dtype=torch.float64)[None]
    t = torch.tensor([[0.8, -0.3, 0.2]], dtype=torch.float64)
    flat = pts_cam2.reshape(1, 3, -1).transpose(1, 2)                   # (1, N, 3)
    in_cam1 = torch.einsum("bij,bnj->bni", R, flat) + t[:, None]        # (1, N, 3)
    pts_in_cam1 = in_cam1.transpose(1, 2).reshape(pts_cam2.shape)       # (1, 3, H, W)

    R_hat, t_hat, s_hat = PM.relative_pose_from_pointmaps(pts_cam2, pts_in_cam1)

    assert torch.allclose(R_hat, R, atol=1e-8)
    assert torch.allclose(t_hat, t, atol=1e-7)
    assert torch.allclose(s_hat, torch.ones(1, dtype=torch.float64), atol=1e-9)


def test_pointmap_net_shapes_and_confidence_range():
    torch.manual_seed(0)
    net = PM.PointmapNet(width=16)
    img1, img2 = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    (p1, c1), (p2, c2) = net(img1, img2)

    for p, c in [(p1, c1), (p2, c2)]:
        assert p.shape == (2, 3, 32, 32)
        assert c.shape == (2, 1, 32, 32)
        assert (c >= 1.0).all()                         # the 1 + softplus floor
    assert PM.pointmap_to_depth(p1).shape == (2, 1, 32, 32)
    assert torch.equal(PM.pointmap_to_depth(p1), p1[:, 2:3])


def test_pointmap_net_can_overfit_one_pair():
    """The gradient path from the loss through cross-attention to both heads works."""
    torch.manual_seed(0)
    net = PM.PointmapNet(width=16)
    img1, img2 = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    gt1 = _synthetic_pointmap(H=32, W=32, seed=5).float()
    gt2 = _synthetic_pointmap(H=32, W=32, seed=6).float()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    losses = []
    for _ in range(60):
        opt.zero_grad()
        (p1, c1), (p2, c2) = net(img1, img2)
        loss = (PM.confidence_weighted_loss(p1, gt1, c1)
                + PM.confidence_weighted_loss(p2, gt2, c2))
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
