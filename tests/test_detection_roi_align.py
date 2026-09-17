import torch

torch.set_num_threads(1)

from mlbook.detection.roi_align import bilinear_at, roi_align, roi_pool


def test_bilinear_at_known_values():
    feat = torch.arange(16.0).reshape(1, 4, 4)
    assert bilinear_at(feat, 1.0, 2.0).item() == 6.0
    assert bilinear_at(feat, 0.5, 0.5).item() == (0 + 1 + 4 + 5) / 4
    assert bilinear_at(feat, -3.0, 0.0).item() == 0.0  # far outside → 0


def test_roi_align_on_linear_ramp_is_exact():
    # For a feature map that is linear in x, bilinear sampling is exact, so each bin's
    # value must equal the ramp evaluated at the bin centre.
    W = 16
    feat = torch.arange(W, dtype=torch.float32).repeat(8, 1)[None]  # (1, 8, 16): value = x
    box = torch.tensor([4.0, 2.0, 12.0, 6.0]) * 4  # image coords, stride 4 → feature coords [4, 12]
    out = roi_align(feat, box, output_size=4, spatial_scale=0.25, sampling_ratio=2)
    bin_w = 8 / 4
    expected = torch.tensor([4 + (j + 0.5) * bin_w for j in range(4)])
    assert torch.allclose(out[0, 0], expected) and torch.allclose(out[0, 3], expected)


def test_roi_align_vs_roi_pool_misalignment():
    # ROIPool snaps a box at feature coord 2.6 to 3; ROIAlign keeps 2.6 — so on a ramp the
    # two outputs differ by the quantisation error while ROIAlign tracks the true box.
    feat = torch.arange(20, dtype=torch.float32).repeat(4, 1)[None]  # (1, 4, 20)
    box = torch.tensor([2.6, 0.0, 10.6, 3.0])  # stride 1
    aligned = roi_align(feat, box, 2, 1.0, sampling_ratio=2)
    pooled = roi_pool(feat, box, 2, 1.0)
    assert torch.allclose(aligned[0, 0], torch.tensor([4.6, 8.6]))
    assert not torch.allclose(aligned, pooled)
    assert pooled[0, 0, 0] >= 2.0  # max-pooled integer cells
