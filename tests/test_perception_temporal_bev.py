import numpy as np
import torch

from mlbook.perception import camera_rig as cr
from mlbook.perception import temporal_bev as tb

EXTENT = (-8.0, 8.0, -8.0, 8.0)  # 16 cells of 1 m


def test_bev_cell_centres():
    c = tb.bev_cell_centres(EXTENT, (16, 16))
    assert c.shape == (16, 16, 2)
    assert torch.allclose(c[0, 0], torch.tensor([-7.5, -7.5]))
    assert torch.allclose(c[15, 0], torch.tensor([7.5, -7.5]))


def test_warp_identity_is_identity():
    prev = torch.randn(2, 3, 16, 16)
    out = tb.warp_bev(prev, torch.eye(4).expand(2, 4, 4), EXTENT)
    assert torch.allclose(out, prev, atol=1e-5)


def test_warp_translation_shifts_by_one_cell():
    prev = torch.zeros(1, 1, 16, 16)
    prev[0, 0, 10, 4] = 1.0  # a static object at x = 2.5, y = -3.5 in the previous frame
    T = torch.tensor(cr.ego_motion_transform(dx=1.0, dy=0.0, dyaw=0.0), dtype=torch.float32).unsqueeze(0)
    out = tb.warp_bev(prev, T, EXTENT)
    # ego moved 1 m forward → the object is now 1 m closer: x = 1.5 → cell 9.
    assert torch.isclose(out[0, 0, 9, 4], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)


def test_warp_rotation_quarter_turn():
    prev = torch.zeros(1, 1, 16, 16)
    prev[0, 0, 8, 12] = 1.0  # object at x = 0.5, y = 4.5 (on the left) in the previous frame
    T = torch.tensor(cr.ego_motion_transform(dx=0.0, dy=0.0, dyaw=np.pi / 2), dtype=torch.float32).unsqueeze(0)
    out = tb.warp_bev(prev, T, EXTENT)
    # ego turned left 90°: what was on the left (y=4.5) is now ahead (x=4.5, y=-0.5) → cell (12, 7).
    assert torch.isclose(out[0, 0, 12, 7], torch.tensor(1.0), atol=1e-4)


def test_temporal_self_attention_shapes_and_mixing():
    torch.manual_seed(0)
    attn = tb.TemporalSelfAttention(c=6)
    cur, prev = torch.randn(2, 6, 4, 5), torch.randn(2, 6, 4, 5)
    out = attn(cur, prev)
    assert out.shape == (2, 6, 4, 5)
    assert not torch.allclose(out, attn(cur, cur))  # the aligned previous frame changes the answer


def test_temporal_bev_fusion_first_frame_and_recurrence():
    torch.manual_seed(0)
    fusion = tb.TemporalBEVFusion(c=4, bev_extent=EXTENT)
    cur = torch.randn(1, 4, 16, 16)
    first = fusion(cur, None, None)
    assert first.shape == (1, 4, 16, 16)
    T = torch.tensor(cr.ego_motion_transform(0.5, 0.0, 0.0), dtype=torch.float32).unsqueeze(0)
    second = fusion(torch.randn(1, 4, 16, 16), first, T)
    assert second.shape == (1, 4, 16, 16) and torch.isfinite(second).all()
