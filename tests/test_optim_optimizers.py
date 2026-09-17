"""One focused test per optimizer class in mlbook.optim.optimizers (select with -k <name>).

Each optimizer is checked (a) for convergence on an ill-conditioned quadratic and
(b) step-for-step against the matching torch.optim class.
"""

import numpy as np
import torch

from mlbook.optim import optimizers as opt

# f(x) = 1/2 x^T A x - b^T x with condition number 25; gradient A x - b.
A = np.diag([1.0, 25.0])
B = np.array([1.0, 1.0])
X_STAR = np.linalg.solve(A, B)


def grad(x):
    return A @ x - B


def _converges(make, n=2000, atol=1e-2):
    x = np.array([3.0, 3.0])
    x_final = opt.run_optimizer(make([x]), grad, n)[-1]
    assert np.allclose(x_final, X_STAR, atol=atol), x_final


def _torch_reference(torch_opt_cls, n_steps, **kwargs):
    x = torch.tensor([3.0, -2.0], dtype=torch.float64, requires_grad=True)
    o = torch_opt_cls([x], **kwargs)
    At, Bt = torch.tensor(A), torch.tensor(B)
    for _ in range(n_steps):
        o.zero_grad()
        (0.5 * x @ At @ x - Bt @ x).backward()
        o.step()
    return x.detach().numpy()


def _run_ours(o, n_steps):
    for _ in range(n_steps):
        o.step([grad(o.params[0])])
    return o.params[0]


def test_sgd():
    _converges(lambda p: opt.SGD(p, lr=0.05))
    x = _run_ours(opt.SGD([np.array([3.0, -2.0])], lr=0.01, weight_decay=0.1), 40)
    assert np.allclose(x, _torch_reference(torch.optim.SGD, 40, lr=0.01, weight_decay=0.1), atol=1e-10)


def test_momentum():
    _converges(lambda p: opt.Momentum(p, lr=0.02, momentum=0.9))
    x = _run_ours(opt.Momentum([np.array([3.0, -2.0])], lr=0.01, momentum=0.9), 40)
    assert np.allclose(x, _torch_reference(torch.optim.SGD, 40, lr=0.01, momentum=0.9), atol=1e-10)


def test_nesterov():
    _converges(lambda p: opt.Nesterov(p, lr=0.02, momentum=0.9))
    x = _run_ours(opt.Nesterov([np.array([3.0, -2.0])], lr=0.01, momentum=0.9), 40)
    ref = _torch_reference(torch.optim.SGD, 40, lr=0.01, momentum=0.9, nesterov=True)
    assert np.allclose(x, ref, atol=1e-10)


def test_adagrad():
    _converges(lambda p: opt.AdaGrad(p, lr=1.0))
    x = _run_ours(opt.AdaGrad([np.array([3.0, -2.0])], lr=0.5, eps=1e-10), 40)
    assert np.allclose(x, _torch_reference(torch.optim.Adagrad, 40, lr=0.5, eps=1e-10), atol=1e-10)


def test_rmsprop():
    _converges(lambda p: opt.RMSProp(p, lr=0.01))
    x = _run_ours(opt.RMSProp([np.array([3.0, -2.0])], lr=0.01, rho=0.99, eps=1e-8), 40)
    ref = _torch_reference(torch.optim.RMSprop, 40, lr=0.01, alpha=0.99, eps=1e-8)
    assert np.allclose(x, ref, atol=1e-10)


def test_adam():
    _converges(lambda p: opt.Adam(p, lr=0.05))
    # exact match with torch.optim.Adam, with and without coupled L2
    x = _run_ours(opt.Adam([np.array([3.0, -2.0])], lr=0.1, betas=(0.9, 0.999), eps=1e-8), 50)
    assert np.allclose(x, _torch_reference(torch.optim.Adam, 50, lr=0.1, betas=(0.9, 0.999), eps=1e-8), atol=1e-10)
    x = _run_ours(opt.Adam([np.array([3.0, -2.0])], lr=0.1, weight_decay=0.1), 30)
    assert np.allclose(x, _torch_reference(torch.optim.Adam, 30, lr=0.1, weight_decay=0.1), atol=1e-10)
    # bias correction: the first step is lr * sign(g) regardless of gradient scale
    x = np.array([1.0, -1.0, 5.0])
    g = np.array([0.001, 10.0, -3.0])
    opt.Adam([x], lr=0.1).step([g])
    assert np.allclose(x, np.array([1.0, -1.0, 5.0]) - 0.1 * np.sign(g), atol=1e-5)


def test_adamw():
    _converges(lambda p: opt.AdamW(p, lr=0.05, weight_decay=0.0))
    x = _run_ours(opt.AdamW([np.array([3.0, -2.0])], lr=0.1, weight_decay=0.1), 30)
    assert np.allclose(x, _torch_reference(torch.optim.AdamW, 30, lr=0.1, weight_decay=0.1), atol=1e-10)
    # decoupled decay is NOT the same as Adam with coupled L2
    ref_adam_l2 = _torch_reference(torch.optim.Adam, 30, lr=0.1, weight_decay=0.1)
    assert not np.allclose(x, ref_adam_l2, atol=1e-3)


def test_run_optimizer():
    traj = opt.run_optimizer(opt.SGD([np.array([3.0, 3.0])], lr=0.05), grad, 10)
    assert traj.shape == (11, 2)
    assert np.allclose(traj[0], [3.0, 3.0])
    assert np.allclose(traj[1], np.array([3.0, 3.0]) - 0.05 * grad(np.array([3.0, 3.0])))
