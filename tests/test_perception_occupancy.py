import torch

from mlbook.perception import camera_rig as cr
from mlbook.perception import occupancy as occ

EXTENT = (-8.0, 8.0, -8.0, 8.0)


def test_occupancy_head_shapes_and_gradient():
    torch.manual_seed(0)
    head = occ.OccupancyHead(c_in=8, num_classes=5, num_z=4)
    logits = head(torch.randn(2, 8, 16, 16))
    assert logits.shape == (2, 5, 4, 16, 16)
    target = torch.randint(0, 5, (2, 4, 16, 16))
    loss = occ.occupancy_loss(logits, target, class_weights=torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss) and head.conv[0].weight.grad is not None


def test_temporal_occupancy_fusion_first_frame_and_gate_range():
    torch.manual_seed(0)
    fusion = occ.TemporalOccupancyFusion(c=4, bev_extent=EXTENT)
    cur = torch.randn(1, 4, 16, 16)
    assert torch.equal(fusion(cur, None, None), cur)
    prev = torch.randn(1, 4, 16, 16)
    out = fusion(cur, prev, torch.eye(4).unsqueeze(0))
    lo, hi = torch.minimum(cur, prev), torch.maximum(cur, prev)
    assert ((out >= lo - 1e-5) & (out <= hi + 1e-5)).all()  # convex combination of the two


def test_temporal_fusion_memory_follows_ego_motion():
    torch.manual_seed(0)
    fusion = occ.TemporalOccupancyFusion(c=1, bev_extent=EXTENT)
    with torch.no_grad():
        fusion.gate.weight.zero_()
        fusion.gate.bias.fill_(-20.0)  # gate ≈ 0 → trust the memory only
    prev = torch.zeros(1, 1, 16, 16)
    prev[0, 0, 12, 8] = 1.0  # occupied cell at x = 4.5 in the previous frame
    T = torch.tensor(cr.ego_motion_transform(2.0, 0.0, 0.0), dtype=torch.float32).unsqueeze(0)
    out = fusion(torch.zeros(1, 1, 16, 16), prev, T)
    assert torch.isclose(out[0, 0, 10, 8], torch.tensor(1.0), atol=1e-4)  # moved 2 cells closer


def test_voxelize_points_and_iou():
    pts = torch.tensor([[0.5, 0.5, 0.5], [-7.5, 7.5, -0.5]])
    labels = torch.tensor([2, 1])
    target = occ.voxelize_points(pts, labels, (-8.0, 8.0, -8.0, 8.0, -1.0, 1.0), grid=(2, 16, 16))
    assert target.shape == (2, 16, 16)
    assert target[1, 8, 8] == 2 and target[0, 0, 15] == 1
    assert (target == 0).sum() == 2 * 16 * 16 - 2
    logits = torch.nn.functional.one_hot(target, 3).permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, 3, Z, X, Y)
    binary, per_class = occ.occupancy_iou(logits, target.unsqueeze(0))
    assert binary == 1.0 and torch.allclose(per_class, torch.ones(3))
