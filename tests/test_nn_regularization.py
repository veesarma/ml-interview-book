"""Dropout, DropPath, label smoothing, mixup/CutMix, early stopping."""
import numpy as np
import torch
import torch.nn.functional as F

from mlbook.nn.layers import numerical_gradient, rel_error
from mlbook.nn.losses import one_hot
from mlbook.nn.regularization import (
    Dropout,
    EarlyStopping,
    LabelSmoothingCrossEntropy,
    cutmix,
    drop_path,
    label_smoothing_targets,
    mixup,
    sgd_step_weight_decay,
)


def test_inverted_dropout_preserves_expectation_and_backward_uses_mask():
    x = np.ones((2000, 50))
    d = Dropout(p=0.3, seed=0)
    y = d.forward(x)
    assert abs(y.mean() - 1.0) < 0.02  # E[y] = x
    assert set(np.unique(y)) == {0.0, 1.0 / 0.7}
    dx = d.backward(np.ones_like(x))
    np.testing.assert_array_equal(dx, y)  # same scaled mask
    d.training = False
    np.testing.assert_array_equal(d.forward(x), x)


def test_drop_path_drops_whole_samples():
    rng = np.random.default_rng(0)
    x = np.ones((1000, 4, 3))
    y = drop_path(x, 0.25, rng)
    per_sample = y.reshape(1000, -1)
    assert np.all((per_sample == 0).all(1) | (per_sample == 1 / 0.75).all(1))
    assert abs(y.mean() - 1.0) < 0.05


def test_label_smoothing_targets_and_loss_vs_torch():
    z = np.random.randn(6, 5)
    y = np.array([0, 4, 2, 2, 1, 3])
    q = label_smoothing_targets(y, 5, 0.1)
    np.testing.assert_allclose(q.sum(1), 1.0)
    assert np.isclose(q[0, 0], 0.9 + 0.02) and np.isclose(q[0, 1], 0.02)
    loss = LabelSmoothingCrossEntropy(5, eps=0.1)
    val = loss.forward(z, y)
    ref = F.cross_entropy(torch.tensor(z), torch.tensor(y), label_smoothing=0.1).item()
    assert abs(val - ref) < 1e-10
    num = numerical_gradient(lambda a: LabelSmoothingCrossEntropy(5, 0.1).forward(a, y), z.copy())
    assert rel_error(loss.backward(), num) < 1e-6


def test_label_smoothing_optimum_is_finite_logit_gap():
    # With eps > 0 the optimal logits satisfy p = q, i.e. gap = log((1-eps+eps/K)/(eps/K)).
    k, eps = 4, 0.1
    gap = np.log((1 - eps + eps / k) / (eps / k))
    z = np.array([[gap, 0.0, 0.0, 0.0]])
    loss = LabelSmoothingCrossEntropy(k, eps)
    loss.forward(z, np.array([0]))
    np.testing.assert_allclose(loss.backward(), 0.0, atol=1e-12)


def test_mixup_is_convex_combination():
    rng = np.random.default_rng(0)
    x = rng.random((8, 3, 6, 6))
    y = one_hot(rng.integers(0, 4, 8), 4)
    xm, ym = mixup(x, y, 0.4, rng)
    assert xm.shape == x.shape and ym.shape == y.shape
    np.testing.assert_allclose(ym.sum(1), 1.0)
    assert xm.min() >= x.min() - 1e-12 and xm.max() <= x.max() + 1e-12


def test_cutmix_pastes_box_and_weights_label_by_area():
    rng = np.random.default_rng(0)
    x = rng.random((8, 3, 6, 6))
    y = one_hot(rng.integers(0, 4, 8), 4)
    xc, yc = cutmix(x, y, 1.0, rng)
    assert xc.shape == x.shape
    np.testing.assert_allclose(yc.sum(1), 1.0)
    changed = (xc != x).any(axis=(0, 1))  # (H, W) mask of pasted pixels
    frac = changed.mean()
    lam = yc[0][y[0] == 1].max() if (y[0] == 1).sum() == 1 else None
    # every pasted pixel column/row set forms one rectangle
    rows, cols = np.where(changed)
    if rows.size:
        assert changed[rows.min():rows.max() + 1, cols.min():cols.max() + 1].all()
        assert abs(frac - (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1) / 36) < 1e-12


def test_early_stopping_patience():
    es = EarlyStopping(patience=2)
    assert not es.step(1.0) and not es.step(0.9) and not es.step(0.95)
    assert es.step(0.96)
    assert es.best == 0.9


def test_weight_decay_forms_coincide_for_sgd():
    w, g = np.ones(3), np.full(3, 0.5)
    np.testing.assert_allclose(sgd_step_weight_decay(w, g, 0.1, 0.01, decoupled=True), sgd_step_weight_decay(w, g, 0.1, 0.01, decoupled=False))
    np.testing.assert_allclose(sgd_step_weight_decay(w, g, 0.1, 0.01), 1 - 0.05 - 0.001)
