"""One focused test per schedule in mlbook.optim.schedules (select with -k <name>)."""

import math

import numpy as np
import torch

from mlbook.optim import schedules as sch


def test_linear_warmup():
    assert np.isclose(sch.linear_warmup(0, 10, 1.0), 0.1)
    assert np.isclose(sch.linear_warmup(9, 10, 1.0), 1.0)
    assert np.isclose(sch.linear_warmup(100, 10, 1.0), 1.0)
    assert sch.linear_warmup(0, 0, 1.0) == 1.0


def test_warmup_cosine():
    peak, total, warm = 3e-4, 1000, 100
    assert np.isclose(sch.warmup_cosine(warm, warm, total, peak), peak)
    assert np.isclose(sch.warmup_cosine(total, warm, total, peak, min_lr=3e-5), 3e-5)
    mid = warm + (total - warm) // 2
    assert np.isclose(sch.warmup_cosine(mid, warm, total, peak), peak / 2)
    assert sch.warmup_cosine(0, warm, total, peak) < sch.warmup_cosine(50, warm, total, peak)
    # post-warmup part equals torch CosineAnnealingLR
    p = torch.nn.Parameter(torch.zeros(1))
    o = torch.optim.SGD([p], lr=peak)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=total - warm, eta_min=0.0)
    for step in range(warm, total + 1):
        assert np.isclose(sch.warmup_cosine(step, warm, total, peak), o.param_groups[0]["lr"], rtol=1e-6)
        o.step()
        cos.step()


def test_warmup_stable_decay():
    f = lambda s: sch.warmup_stable_decay(s, 10, 80, 10, 1.0, min_lr=0.1)  # noqa: E731
    assert f(0) < f(9) <= f(10) == f(50) == f(89) == 1.0
    assert f(91) < 1.0 and np.isclose(f(95), 0.55) and np.isclose(f(100), 0.1) and np.isclose(f(1000), 0.1)


def test_inverse_sqrt():
    lrs = [sch.inverse_sqrt(s, 4000, 1.0) for s in range(0, 20000)]
    assert np.argmax(lrs) == 3999 and np.isclose(max(lrs), 1.0)
    assert np.isclose(sch.inverse_sqrt(15999, 4000, 1.0), math.sqrt(4000 / 16000))


def test_step_decay():
    assert sch.step_decay(0, 0.1, 30) == 0.1
    assert np.isclose(sch.step_decay(30, 0.1, 30), 0.01)
    assert np.isclose(sch.step_decay(65, 0.1, 30), 0.001)
