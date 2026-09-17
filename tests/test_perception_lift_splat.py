import numpy as np
import torch

from mlbook.perception import camera_rig as cr
from mlbook.perception import lift_splat as lss


def _rig_tensors(n_cams=6, image_hw=(128, 224)):
    cams = cr.make_surround_rig(n_cams=n_cams, fov_x_deg=70.0, image_hw=image_hw)
    K = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32)
    T = torch.tensor(np.stack([c.T_ego_from_cam for c in cams]), dtype=torch.float32)
    return cams, K, T


def test_create_frustum_shape_and_pixel_centres():
    fr = lss.create_frustum((128, 224), (8, 14), torch.tensor([2.0, 4.0, 6.0]))
    assert fr.shape == (3, 8, 14, 3)
    assert torch.isclose(fr[0, 0, 0, 0], torch.tensor(8.0))  # first column centre: 0.5 · (224/14)
    assert torch.isclose(fr[0, 0, 0, 1], torch.tensor(8.0))  # first row centre: 0.5 · (128/8)
    assert torch.equal(fr[:, 3, 5, 2], torch.tensor([2.0, 4.0, 6.0]))


def test_frustum_to_ego_matches_numpy_camera_unproject():
    cams, K, T = _rig_tensors()
    fr = lss.create_frustum((128, 224), (8, 14), torch.linspace(2.0, 30.0, 5))
    p_ego = lss.frustum_to_ego(fr, K, T)  # (N, D, Hf, Wf, 3)
    assert p_ego.shape == (6, 5, 8, 14, 3)
    for i, cam in enumerate(cams):
        flat = fr.reshape(-1, 3).numpy().astype(np.float64)
        ref = cam.unproject(flat[:, :2], flat[:, 2])
        assert np.allclose(p_ego[i].reshape(-1, 3).numpy(), ref, atol=1e-3)


def test_pillar_pool_puts_feature_in_expected_cell():
    extent, hw = (-10.0, 10.0, -10.0, 10.0), (20, 20)
    pts = torch.zeros(1, 1, 1, 2, 3)  # (N=1, D=1, Hf=1, Wf=2, 3)
    pts[0, 0, 0, 0] = torch.tensor([3.5, -2.5, 0.0])  # → cell ix = 13, iy = 7
    pts[0, 0, 0, 1] = torch.tensor([50.0, 0.0, 0.0])  # outside → dropped
    feats = torch.ones(2, 1, 1, 1, 2, 4)  # (B=2, N, D, Hf, Wf, C=4)
    bev = lss.pillar_pool(feats, pts, extent, hw)
    assert bev.shape == (2, 4, 20, 20)
    assert torch.allclose(bev[:, :, 13, 7], torch.ones(2, 4))
    assert torch.isclose(bev.sum(), torch.tensor(8.0))  # 2 batches × 4 channels × 1 kept point


def test_pillar_pool_is_differentiable_and_sums_duplicates():
    pts = torch.zeros(1, 1, 1, 3, 3)  # three frustum points in the same cell
    feats = torch.randn(1, 1, 1, 1, 3, 2, requires_grad=True)
    bev = lss.pillar_pool(feats, pts, (-1.0, 1.0, -1.0, 1.0), (2, 2))
    assert torch.allclose(bev[0, :, 1, 1], feats[0, 0, 0, 0].sum(0))
    bev.sum().backward()
    assert torch.allclose(feats.grad, torch.ones_like(feats))


def test_lift_splat_end_to_end_shapes_and_depth_distribution():
    torch.manual_seed(0)
    _, K, T = _rig_tensors()
    depth_bins = torch.linspace(2.0, 40.0, 8)
    model = lss.LiftSplat(c_in=16, c_out=8, depth_bins=depth_bins, image_hw=(128, 224), feature_hw=(8, 14),
                          bev_extent=(-40.0, 40.0, -40.0, 40.0), bev_hw=(32, 32))
    feats = torch.randn(2, 6, 16, 8, 14)
    bev, depth_prob = model(feats, K, T)
    assert bev.shape == (2, 8, 32, 32)
    assert depth_prob.shape == (2, 6, 8, 8, 14)
    assert torch.allclose(depth_prob.sum(2), torch.ones(2, 6, 8, 14), atol=1e-5)
    assert bev.abs().sum() > 0
    bev.sum().backward()
    assert model.depth_head.weight.grad is not None


def test_depth_bin_targets_nearest_bin_and_ignore():
    bins = torch.tensor([2.0, 4.0, 6.0])
    d = torch.tensor([[[[0.0, 2.9], [5.2, 100.0]]]])  # (B=1, N=1, 2, 2)
    t = lss.depth_bin_targets(d, bins)
    assert t.tolist() == [[[[-1, 0], [2, 2]]]]
