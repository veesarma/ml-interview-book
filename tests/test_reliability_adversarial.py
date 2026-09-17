import copy
import functools

import torch

from mlbook.reliability import adversarial as adv

torch.set_num_threads(1)


@functools.lru_cache(maxsize=1)
def _cached():
    torch.manual_seed(0)
    x = torch.rand(200, 2)
    y = (x[:, 0] + x[:, 1] > 1.0).long()
    model = adv.TinyMLP(2, 16, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(60):
        opt.zero_grad(); torch.nn.functional.cross_entropy(model(x), y).backward(); opt.step()
    return model, x, y


def _trained_model():
    model, x, y = _cached()
    return copy.deepcopy(model), x, y


def test_input_gradient_matches_finite_difference():
    model, x, y = _trained_model()
    g = adv.input_gradient(model, x[:1], y[:1])
    eps = 1e-3
    for j in range(2):
        xp, xm = x[:1].clone(), x[:1].clone()
        xp[0, j] += eps; xm[0, j] -= eps
        fd = (torch.nn.functional.cross_entropy(model(xp), y[:1]) - torch.nn.functional.cross_entropy(model(xm), y[:1])) / (2 * eps)
        assert torch.isclose(g[0, j], fd, atol=1e-3)


def test_fgsm_increases_loss_and_respects_eps():
    model, x, y = _trained_model()
    x_adv = adv.fgsm(model, x, y, eps=0.05)
    assert (x_adv - x).abs().max() <= 0.05 + 1e-6 and x_adv.min() >= 0 and x_adv.max() <= 1
    ce = torch.nn.functional.cross_entropy
    assert ce(model(x_adv), y) > ce(model(x), y)
    acc_clean = (model(x).argmax(1) == y).float().mean()
    acc_adv = (model(x_adv).argmax(1) == y).float().mean()
    assert acc_adv < acc_clean


def test_pgd_stronger_than_fgsm_and_within_ball():
    model, x, y = _trained_model()
    x_f = adv.fgsm(model, x, y, eps=0.08)
    x_p = adv.pgd(model, x, y, eps=0.08, alpha=0.02, n_steps=10)
    assert (x_p - x).abs().max() <= 0.08 + 1e-6
    ce = torch.nn.functional.cross_entropy
    assert ce(model(x_p), y) >= ce(model(x_f), y) - 1e-4


def test_adversarial_training_step_improves_robust_loss():
    model, x, y = _trained_model()
    ce = torch.nn.functional.cross_entropy
    robust_loss = lambda m: float(ce(m(adv.pgd(m, x, y, 0.08, 0.02, 5, random_start=False)), y))  # noqa: E731
    before = robust_loss(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(15):
        step_loss = adv.adversarial_training_step(model, opt, x, y, 0.08, 0.02, 3)
        assert step_loss > 0
    assert robust_loss(model) < before
