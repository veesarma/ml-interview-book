import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mlbook.multimodal.detr_loss import detr_loss, hungarian_match, matching_as_permutation, pairwise_giou
from mlbook.multimodal.hungarian import assignment_cost, hungarian

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers


def test_hungarian_matches_scipy_square_and_rectangular():
    rng = np.random.default_rng(0)
    for n, m in [(1, 1), (3, 3), (5, 5), (4, 7), (7, 4), (10, 10), (2, 9)]:
        for _ in range(5):
            C = rng.random((n, m))
            r, c = hungarian(C)
            r2, c2 = linear_sum_assignment(C)
            assert len(r) == min(n, m)
            assert len(set(c.tolist())) == len(c) and len(set(r.tolist())) == len(r)
            assert np.isclose(assignment_cost(C, r, c), C[r2, c2].sum())


def test_hungarian_known_answer():
    C = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
    r, c = hungarian(C)
    assert assignment_cost(C, r, c) == 5.0  # 1 + 2 + 2


def test_giou_known_values():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 2.0, 1.0], [2.0, 0.0, 3.0, 1.0]])
    g = pairwise_giou(a, b)[0]
    assert torch.isclose(g[0], torch.tensor(1.0))
    assert torch.isclose(g[1], torch.tensor(0.0))  # touching: IoU 0, enclosing = union
    assert torch.isclose(g[2], torch.tensor(-1.0 / 3.0))  # gap of 1 in enclosing width 3


def _perfect_case():
    K = 3
    boxes_t = torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.3, 0.4]])
    labels_t = torch.tensor([1, 2])
    Q = 4
    logits = torch.full((1, Q, K + 1), -30.0)
    logits[0, 0, 1] = 30.0
    logits[0, 1, 2] = 30.0
    logits[0, 2, K] = 30.0
    logits[0, 3, K] = 30.0
    boxes = torch.rand(1, Q, 4) * 0.2 + 0.4
    boxes[0, 0] = boxes_t[0]
    boxes[0, 1] = boxes_t[1]
    return logits, boxes, [{"labels": labels_t, "boxes": boxes_t}]


def test_loss_zero_for_perfect_prediction():
    logits, boxes, targets = _perfect_case()
    out = detr_loss(logits, boxes, targets)
    assert out["loss_l1"].item() < 1e-6
    assert out["loss_giou"].item() < 1e-6
    assert out["loss_ce"].item() < 1e-6
    assert out["loss"].item() < 1e-5


def test_matching_permutation_invariant():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 5)
    boxes = torch.rand(1, 6, 4) * 0.5 + 0.25
    targets = [{"labels": torch.tensor([0, 3, 1]), "boxes": torch.rand(3, 4) * 0.5 + 0.25}]
    perm_q = torch.tensor([5, 2, 0, 3, 1, 4])
    match_a = matching_as_permutation(hungarian_match(logits, boxes, targets), 6)[0]
    match_b = matching_as_permutation(hungarian_match(logits[:, perm_q], boxes[:, perm_q], targets), 6)[0]
    # query perm_q[i] in the original == query i in the permuted problem
    assert np.array_equal(match_a[perm_q.numpy()], match_b)
    # permuting the targets also leaves the loss unchanged
    perm_t = torch.tensor([2, 0, 1])
    t2 = [{"labels": targets[0]["labels"][perm_t], "boxes": targets[0]["boxes"][perm_t]}]
    la = detr_loss(logits, boxes, targets)["loss"]
    lb = detr_loss(logits[:, perm_q], boxes[:, perm_q], t2)["loss"]
    assert torch.isclose(la, lb, atol=1e-6)


def test_loss_backward_and_empty_targets():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    boxes = (torch.rand(2, 5, 4) * 0.5 + 0.25).requires_grad_()
    targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 4) * 0.5 + 0.25},
               {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}]
    out = detr_loss(logits, boxes, targets)
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(boxes.grad).all()
