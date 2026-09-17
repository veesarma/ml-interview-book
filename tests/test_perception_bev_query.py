import numpy as np
import torch

from mlbook.perception import bev_query as bq
from mlbook.perception import camera_rig as cr


def _rig(n_cams=6, image_hw=(128, 224)):
    cams = cr.make_surround_rig(n_cams=n_cams, fov_x_deg=70.0, image_hw=image_hw)
    K = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32)
    T = torch.tensor(np.stack([c.T_cam_from_ego for c in cams]), dtype=torch.float32)
    return cams, K, T


def test_bev_reference_points_layout():
    ref = bq.bev_reference_points((-10.0, 10.0, -10.0, 10.0), (4, 4), torch.tensor([-1.0, 0.0, 1.0]))
    assert ref.shape == (16, 3, 3)
    assert torch.allclose(ref[0, :, :2], torch.tensor([[-7.5, -7.5]] * 3))
    assert torch.equal(ref[0, :, 2], torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.allclose(ref[1, 0, :2], torch.tensor([-7.5, -2.5]))  # Y is the fast axis


def test_project_points_to_cameras_matches_numpy_camera():
    cams, K, T = _rig()
    pts = torch.tensor(np.random.default_rng(0).uniform(-20, 20, size=(200, 3)), dtype=torch.float32)
    pix, valid = bq.project_points_to_cameras(pts, K, T, (128, 224))
    assert pix.shape == (6, 200, 2) and valid.shape == (6, 200)
    for i, cam in enumerate(cams):
        ref_pix, _, ref_valid = cam.project(pts.numpy().astype(np.float64))
        assert torch.equal(valid[i], torch.tensor(ref_valid))
        assert np.allclose(pix[i][valid[i]].numpy(), ref_pix[ref_valid], atol=1e-2)


def test_query_attends_only_to_the_camera_that_sees_it():
    torch.manual_seed(0)
    _, K, T = _rig()
    attn = bq.BEVQueryCrossAttention(d_model=8, c_img=4, image_hw=(128, 224))
    # One query straight ahead of cam0 (x=12, y=0) and one far behind (x=-12): cam0 vs cam3.
    ref = torch.tensor([[[12.0, 0.0, 0.5]], [[-12.0, 0.0, 0.5]]])  # (Nq=2, N_z=1, 3)
    feats = torch.zeros(1, 6, 4, 8, 14)
    feats[:, 0] = 1.0  # cam0 features are all ones, every other camera is zero
    sampled, valid = attn.sample_image_features(feats, ref, K, T)
    assert sampled.shape == (1, 2, 6, 4) and valid.shape == (2, 6)
    assert valid[0].tolist() == [True, False, False, False, False, False]
    assert valid[1].tolist() == [False, False, False, True, False, False]
    assert torch.allclose(sampled[0, 0, 0], torch.ones(4))  # the front query reads cam0's ones
    assert torch.allclose(sampled[0, 1, 3], torch.zeros(4))


def test_query_with_no_hits_is_unchanged_and_others_get_gradient():
    torch.manual_seed(0)
    _, K, T = _rig()
    attn = bq.BEVQueryCrossAttention(d_model=8, c_img=4, image_hw=(128, 224))
    ref = torch.tensor([[[12.0, 0.0, 0.5]], [[0.0, 0.0, 30.0]]])  # second pillar is 30 m up: no camera sees it
    q = torch.randn(1, 2, 8, requires_grad=True)
    feats = torch.randn(1, 6, 4, 8, 14)
    out = attn(q, feats, ref, K, T)
    assert out.shape == (1, 2, 8)
    assert torch.allclose(out[0, 1], q[0, 1])
    assert not torch.allclose(out[0, 0], q[0, 0])
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_petr_position_encoder_shape_and_view_consistency():
    torch.manual_seed(0)
    cams = cr.make_surround_rig(n_cams=2, fov_x_deg=70.0, image_hw=(128, 224))
    K = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32)
    T = torch.tensor(np.stack([c.T_ego_from_cam for c in cams]), dtype=torch.float32)
    enc = bq.PETRPositionEncoder(d_model=16, image_hw=(128, 224), feature_hw=(8, 14), depth_bins=torch.linspace(1, 30, 4),
                                 extent=(-40.0, 40.0, -40.0, 40.0, -3.0, 5.0))
    pe = enc(K, T)
    assert pe.shape == (2, 8, 14, 16)
    # The two cameras face opposite ways, so the same pixel must get different 3D embeddings.
    assert not torch.allclose(pe[0, 4, 7], pe[1, 4, 7])
