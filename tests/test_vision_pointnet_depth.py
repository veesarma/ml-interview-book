import numpy as np
import torch
import torch.nn.functional as F

from mlbook.vision import depth as D
from mlbook.vision.pointnet import TinyPointNet, feature_transform_regularizer, synthetic_point_clouds


def test_pointnet_permutation_invariance_and_learning():
    pts, y = synthetic_point_clouds(48, num_points=64)
    model = TinyPointNet(4, width=16)
    model.eval()
    with torch.no_grad():
        logits, _ = model(pts)
        perm = torch.randperm(64)
        logits_perm, _ = model(pts[:, perm])
    assert torch.allclose(logits, logits_perm, atol=1e-5)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(60):
        opt.zero_grad()
        out, T = model(pts)
        (F.cross_entropy(out, y) + 1e-3 * feature_transform_regularizer(T)).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(pts)[0].argmax(1) == y).float().mean().item()
    assert acc > 0.85


def test_disparity_depth_roundtrip_and_error_growth():
    f, b = 700.0, 0.54  # KITTI-like
    d = np.array([70.0, 7.0, 3.5])
    z = D.disparity_to_depth(d, f, b)
    assert np.allclose(z, [5.4, 54.0, 108.0])
    assert np.allclose(D.depth_to_disparity(z, f, b), d)
    err = D.depth_error_from_disparity_error(z, f, b, disp_err_px=0.5)
    assert err[1] / err[0] == 100.0  # 10× range → 100× depth error


def test_block_matching_recovers_constant_shift():
    rng = np.random.default_rng(0)
    left = rng.random((24, 40))
    left = 0.5 * left + 0.5 * np.roll(left, 1, axis=1)  # a little spatial correlation
    shift = 4
    right = np.roll(left, -shift, axis=1)  # right image = left shifted left by 4 → disparity 4
    disp = D.stereo_block_matching(left, right, max_disparity=8, block=5)
    interior = disp[3:-3, 10:-3]
    assert (interior == shift).mean() > 0.9


def test_warp_and_photometric_loss_zero_for_true_disparity():
    torch.manual_seed(0)
    right = torch.rand(1, 1, 8, 16)
    right = F.avg_pool2d(right, 3, 1, 1)
    d = 3.0
    left = torch.zeros_like(right)
    left[..., int(d):] = right[..., : 16 - int(d)]  # left(x) = right(x − d)
    rec = D.warp_right_to_left(right, torch.full((1, 8, 16), d))
    assert torch.allclose(rec[..., 3:], left[..., 3:], atol=1e-6)
    assert D.photometric_loss(rec[..., 4:], left[..., 4:]).item() < 1e-5
    vol = D.build_cost_volume(right, right, max_disparity=4)
    assert vol.shape == (1, 2, 4, 8, 16)
    cost = torch.rand(1, 4, 8, 16)
    cost[:, 2] = -10.0
    assert torch.allclose(D.soft_argmin_disparity(cost), torch.full((1, 8, 16), 2.0), atol=1e-3)
    assert D.smoothness_loss(torch.full((1, 8, 16), 1.0), right).item() == 0.0
