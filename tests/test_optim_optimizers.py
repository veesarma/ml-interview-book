import numpy as np
import torch

from mlbook.optim import optimizers as opt

# Ill-conditioned quadratic f(x) = 1/2 x^T A x - b^T x, gradient A x - b.
A = np.diag([1.0, 25.0])
B = np.array([1.0, 1.0])
X_STAR = np.linalg.solve(A, B)


def grad(x):
    return A @ x - B


def _run(optimizer, n=500):
    return opt.run_optimizer(optimizer, grad, n)[-1]


def test_all_optimizers_converge_on_quadratic():
    for make in [
        lambda p: opt.SGD(p, lr=0.05),
        lambda p: opt.Momentum(p, lr=0.02, momentum=0.9),
        lambda p: opt.Nesterov(p, lr=0.02, momentum=0.9),
        lambda p: opt.AdaGrad(p, lr=1.0),
        lambda p: opt.RMSProp(p, lr=0.01),
        lambda p: opt.Adam(p, lr=0.05),
        lambda p: opt.AdamW(p, lr=0.05, weight_decay=0.0),
    ]:
        x = np.array([3.0, 3.0])
        x_final = _run(make([x]), n=2000)
        assert np.allclose(x_final, X_STAR, atol=1e-2), make


def _torch_reference(torch_opt_cls, n_steps, **kwargs):
    x = torch.tensor([3.0, -2.0], dtype=torch.float64, requires_grad=True)
    o = torch_opt_cls([x], **kwargs)
    At, Bt = torch.tensor(A), torch.tensor(B)
    for _ in range(n_steps):
        o.zero_grad()
        loss = 0.5 * x @ At @ x - Bt @ x
        loss.backward()
        o.step()
    return x.detach().numpy()


def test_adam_matches_torch_adam_exactly():
    x = np.array([3.0, -2.0])
    o = opt.Adam([x], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(50):
        o.step([grad(x)])
    ref = _torch_reference(torch.optim.Adam, 50, lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    assert np.allclose(x, ref, atol=1e-10)


def test_adam_coupled_l2_matches_torch():
    x = np.array([3.0, -2.0])
    o = opt.Adam([x], lr=0.1, weight_decay=0.1)
    for _ in range(30):
        o.step([grad(x)])
    ref = _torch_reference(torch.optim.Adam, 30, lr=0.1, weight_decay=0.1)
    assert np.allclose(x, ref, atol=1e-10)


def test_adamw_matches_torch_adamw_and_differs_from_adam_l2():
    x = np.array([3.0, -2.0])
    o = opt.AdamW([x], lr=0.1, weight_decay=0.1)
    for _ in range(30):
        o.step([grad(x)])
    ref = _torch_reference(torch.optim.AdamW, 30, lr=0.1, weight_decay=0.1)
    assert np.allclose(x, ref, atol=1e-10)
    ref_adam_l2 = _torch_reference(torch.optim.Adam, 30, lr=0.1, weight_decay=0.1)
    assert not np.allclose(x, ref_adam_l2, atol=1e-3)


def test_sgd_momentum_nesterov_match_torch():
    for cls, torch_kwargs in [
        (opt.Momentum, dict(momentum=0.9)),
        (opt.Nesterov, dict(momentum=0.9, nesterov=True)),
    ]:
        x = np.array([3.0, -2.0])
        o = cls([x], lr=0.01, momentum=0.9)
        for _ in range(40):
            o.step([grad(x)])
        ref = _torch_reference(torch.optim.SGD, 40, lr=0.01, **torch_kwargs)
        assert np.allclose(x, ref, atol=1e-10), cls


def test_rmsprop_and_adagrad_match_torch():
    x = np.array([3.0, -2.0])
    o = opt.RMSProp([x], lr=0.01, rho=0.99, eps=1e-8)
    for _ in range(40):
        o.step([grad(x)])
    ref = _torch_reference(torch.optim.RMSprop, 40, lr=0.01, alpha=0.99, eps=1e-8)
    assert np.allclose(x, ref, atol=1e-10)

    x = np.array([3.0, -2.0])
    o = opt.AdaGrad([x], lr=0.5, eps=1e-10)
    for _ in range(40):
        o.step([grad(x)])
    ref = _torch_reference(torch.optim.Adagrad, 40, lr=0.5, eps=1e-10)
    assert np.allclose(x, ref, atol=1e-10)


def test_adam_bias_correction_first_step_is_sign_step():
    # With bias correction, the first Adam step is lr * g / (|g| + eps) ~ lr * sign(g).
    x = np.array([1.0, -1.0, 5.0])
    g = np.array([0.001, 10.0, -3.0])
    o = opt.Adam([x], lr=0.1)
    o.step([g])
    assert np.allclose(x, np.array([1.0, -1.0, 5.0]) - 0.1 * np.sign(g), atol=1e-5)
