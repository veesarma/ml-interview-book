"""Pseudo-labelling, entropy minimisation, FixMatch and Mean-Teacher consistency."""
import math

import torch

torch.set_num_threads(1)

import torch.nn.functional as F

from mlbook.ssl import fixmatch as FX
from mlbook.ssl import pseudo_label as PL


# ------------------------------------------------------------------ pseudo_label.py
def test_confident_pseudo_labels_threshold_and_argmax():
    logits = torch.tensor([[5.0, 0.0, 0.0],      # p_max = 0.987  accepted at both thresholds
                           [0.2, 0.1, 0.0],      # p_max = 0.367  rejected at both
                           [0.0, 3.0, 0.0]])     # p_max = 0.909  accepted at 0.90, rejected at 0.95
    pseudo, mask = PL.confident_pseudo_labels(logits, threshold=0.90)
    assert pseudo.tolist() == [0, 0, 1]
    assert mask.tolist() == [1.0, 0.0, 1.0]
    _, mask_strict = PL.confident_pseudo_labels(logits, threshold=0.95)
    assert mask_strict.tolist() == [1.0, 0.0, 0.0]


def test_pseudo_labels_carry_no_gradient_into_the_target():
    logits = torch.randn(8, 4, requires_grad=True)
    pseudo, mask = PL.confident_pseudo_labels(logits, 0.0)
    assert not pseudo.requires_grad and not mask.requires_grad


def test_pseudo_label_loss_is_averaged_over_accepted_examples_only():
    logits = torch.tensor([[10.0, 0.0], [0.05, 0.0]])   # one confident, one not
    loss, frac = PL.pseudo_label_loss(logits, threshold=0.9)
    assert torch.isclose(frac, torch.tensor(0.5))
    manual = F.cross_entropy(logits[:1], torch.tensor([0]))
    assert torch.isclose(loss, manual)


def test_pseudo_label_loss_is_zero_when_nothing_passes_the_threshold():
    loss, frac = PL.pseudo_label_loss(torch.zeros(4, 3), threshold=0.99)
    assert loss.item() == 0.0 and frac.item() == 0.0


def test_pseudo_label_loss_sharpens_predictions():
    logits = torch.tensor([[3.0, 0.0]], requires_grad=True)
    loss, _ = PL.pseudo_label_loss(logits, threshold=0.5)
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0     # push further toward the argmax


def test_semi_supervised_loss_combines_terms():
    logits_l = torch.tensor([[4.0, 0.0]]); y = torch.tensor([0])
    logits_u = torch.tensor([[10.0, 0.0]])
    total, frac = PL.semi_supervised_loss(logits_l, y, logits_u, threshold=0.9, lambda_u=2.0)
    expected = F.cross_entropy(logits_l, y) + 2.0 * PL.pseudo_label_loss(logits_u, 0.9)[0]
    assert torch.isclose(total, expected) and frac.item() == 1.0


def test_entropy_minimization_loss_endpoints():
    uniform = torch.zeros(1, 4)
    assert torch.isclose(PL.entropy_minimization_loss(uniform), torch.tensor(math.log(4.0)))
    peaked = torch.tensor([[50.0, 0.0, 0.0, 0.0]])
    assert PL.entropy_minimization_loss(peaked).item() < 1e-6


def test_class_balance_of_pseudo_labels_flags_imbalance():
    pseudo = torch.tensor([0, 0, 0, 1])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])         # the only class-1 example was rejected
    hist = PL.class_balance_of_pseudo_labels(pseudo, mask, n_classes=2)
    assert hist.tolist() == [3.0, 0.0]                # confirmation bias, visible in one line


# ------------------------------------------------------------------ fixmatch.py
def test_fixmatch_uses_weak_view_for_targets_and_strong_view_for_gradients():
    logits_weak = torch.tensor([[10.0, 0.0]], requires_grad=True)
    logits_strong = torch.tensor([[0.5, 0.4]], requires_grad=True)
    loss, frac = FX.fixmatch_unlabeled_loss(logits_weak, logits_strong, threshold=0.95)
    loss.backward()
    assert frac.item() == 1.0
    assert logits_weak.grad is None                      # the weak branch is detached
    assert logits_strong.grad is not None


def test_fixmatch_divides_by_the_full_unlabeled_batch_not_the_accepted_count():
    """FixMatch's 1/(mu B) normaliser is why the unlabelled loss ramps up as confidence grows."""
    weak = torch.tensor([[10.0, 0.0], [0.0, 0.0]])       # one confident, one at chance
    strong = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    loss, frac = FX.fixmatch_unlabeled_loss(weak, strong, threshold=0.95)
    assert torch.isclose(frac, torch.tensor(0.5))
    assert torch.isclose(loss, torch.tensor(0.5 * math.log(2.0)))   # ½ · CE, not 1 · CE


def test_fixmatch_full_loss_matches_its_parts():
    ll, y = torch.randn(4, 3), torch.randint(0, 3, (4,))
    lw, ls = torch.randn(6, 3), torch.randn(6, 3)
    total, frac = FX.fixmatch_loss(ll, y, lw, ls, threshold=0.5, lambda_u=1.5)
    expected = F.cross_entropy(ll, y) + 1.5 * FX.fixmatch_unlabeled_loss(lw, ls, 0.5)[0]
    assert torch.isclose(total, expected)


def test_augmentations_differ_in_strength():
    x = torch.ones(2000, 4)
    w, s = FX.weak_augment(x), FX.strong_augment(x)
    assert (w - x).abs().mean() < (s - x).abs().mean()
    assert (s == 0).float().mean().item() > 0.1          # strong augmentation drops coordinates


def test_mean_teacher_consistency_endpoints():
    logits = torch.randn(4, 3)
    assert FX.mean_teacher_consistency(logits, logits.clone()).item() < 1e-6
    a = torch.tensor([[50.0, 0.0]]); b = torch.tensor([[0.0, 50.0]])
    assert torch.isclose(FX.mean_teacher_consistency(a, b), torch.tensor(2.0), atol=1e-4)


def test_fixmatch_beats_supervised_only_on_two_clusters_with_few_labels():
    """Two well-separated clusters, 2 labels, 200 unlabelled points: consistency + thresholding
    should propagate the labels to the clusters."""
    torch.manual_seed(0)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    y_u = torch.randint(0, 2, (200,))
    x_u = centers[y_u] + 0.5 * torch.randn(200, 2)
    x_l = centers.clone(); y_l = torch.tensor([0, 1])           # exactly one label per class

    def train(use_fixmatch: bool):
        torch.manual_seed(1)
        net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(300):
            if use_fixmatch:
                loss, _ = FX.fixmatch_loss(net(x_l), y_l, net(FX.weak_augment(x_u, 0.1)),
                                           net(FX.strong_augment(x_u, 0.5, 0.0)), threshold=0.95, lambda_u=1.0)
            else:
                loss = F.cross_entropy(net(x_l), y_l)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return (net(x_u).argmax(dim=1) == y_u).float().mean().item()

    acc_fix, acc_sup = train(True), train(False)
    assert acc_fix >= acc_sup
    assert acc_fix > 0.9
