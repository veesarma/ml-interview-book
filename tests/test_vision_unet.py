import torch

torch.set_num_threads(1)  # the sandbox oversubscribes cores; 1 thread is fastest for these tiny nets

from mlbook.vision.unet import (TinyUNet, boundary_weight_map, dice_loss, mean_iou, panoptic_quality,
                                segmentation_loss, synthetic_segmentation)


def test_unet_shapes():
    m = TinyUNet(1, 3, width=4)
    assert m(torch.randn(2, 1, 32, 32)).shape == (2, 3, 32, 32)


def test_unet_learns_synthetic_shapes():
    x, y = synthetic_segmentation(24)
    m = TinyUNet(1, 3, width=8)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(80):
        opt.zero_grad()
        segmentation_loss(m(x), y, dice_weight=1.0, boundary_weight=3.0).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(x).argmax(1)
    assert mean_iou(pred, y, 3).item() > 0.8


def test_dice_and_miou_known_values():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2] = 1
    perfect = torch.full((1, 2, 4, 4), -20.0)
    perfect[0, 1, :2] = 20.0
    perfect[0, 0, 2:] = 20.0
    assert dice_loss(perfect, target, eps=0.0).item() < 1e-6
    pred = target.clone()
    pred[0, 0, 0] = 0  # one of 8 class-1 pixels wrong
    # class 1: TP=7, FN=1, FP=0 → 7/8; class 0: TP=8, FP=1 → 8/9
    assert abs(mean_iou(pred, target, 2).item() - (7 / 8 + 8 / 9) / 2) < 1e-9


def test_boundary_weights_mark_edges_only():
    target = torch.zeros(1, 6, 6, dtype=torch.long)
    target[0, :, 3:] = 1
    w = boundary_weight_map(target, width=1, boundary_weight=5.0)
    assert w[0, 0, 0].item() == 1.0 and w[0, 0, 2].item() == 5.0 and w[0, 0, 3].item() == 5.0


def test_panoptic_quality():
    g = torch.zeros(8, 8, dtype=torch.bool); g[:4, :4] = True
    p = torch.zeros(8, 8, dtype=torch.bool); p[:4, :2] = True  # IoU = 8/16 = 0.5 → not > 0.5, no match
    assert panoptic_quality([(0, p)], [(0, g)]) == (0.0, 0.0, 0.0)
    p2 = torch.zeros(8, 8, dtype=torch.bool); p2[:4, :3] = True  # IoU 12/16 = 0.75
    extra = torch.zeros(8, 8, dtype=torch.bool); extra[6:, 6:] = True  # false positive
    pq, sq, rq = panoptic_quality([(0, p2), (0, extra)], [(0, g)])
    assert abs(sq - 0.75) < 1e-9 and abs(rq - 1 / 1.5) < 1e-9 and abs(pq - 0.5) < 1e-9
