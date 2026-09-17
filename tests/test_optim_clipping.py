import numpy as np
import torch

from mlbook.optim import clipping as cl


def test_clip_grad_norm_matches_torch():
    grads = [np.random.randn(3, 4) * 5, np.random.randn(7) * 5]
    tg = [torch.tensor(g, requires_grad=True) for g in grads]
    for t, g in zip(tg, grads):
        t.grad = torch.tensor(g)
    total_ref = torch.nn.utils.clip_grad_norm_(tg, max_norm=1.0).item()
    total = cl.clip_grad_norm(grads, max_norm=1.0)
    assert np.isclose(total, total_ref)
    assert np.isclose(cl.global_grad_norm(grads), 1.0)
    for t, g in zip(tg, grads):
        assert np.allclose(t.grad.numpy(), g, atol=1e-6)


def test_clip_preserves_direction_and_noop_below_threshold():
    g0 = np.array([3.0, 4.0])
    g = [g0.copy()]
    cl.clip_grad_norm(g, max_norm=1.0)
    assert np.allclose(g[0], g0 / 5.0, atol=1e-6)
    g = [g0.copy()]
    cl.clip_grad_norm(g, max_norm=10.0)
    assert np.allclose(g[0], g0)


def test_clip_grad_value():
    g = [np.array([-5.0, 0.5, 5.0])]
    cl.clip_grad_value(g, 1.0)
    assert np.allclose(g[0], [-1.0, 0.5, 1.0])
