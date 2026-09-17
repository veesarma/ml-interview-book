import numpy as np
import torch

from mlbook.math import calculus as calc


def test_numerical_gradient_on_known_function():
    f = lambda x: float(np.sum(x**3))  # noqa: E731
    x = np.random.randn(5)
    g = calc.numerical_gradient(f, x)
    assert np.allclose(g, 3 * x**2, atol=1e-6)


def test_grad_quadratic_form_vs_finite_difference():
    A = np.random.randn(4, 4)  # deliberately non-symmetric
    x = np.random.randn(4)
    g_analytic = calc.grad_quadratic_form(A, x)
    g_numeric = calc.numerical_gradient(lambda z: float(z @ A @ z), x)
    assert calc.relative_error(g_analytic, g_numeric) < 1e-6


def test_grad_linear_least_squares_vs_finite_difference_and_torch():
    X, W, Y = np.random.randn(10, 3), np.random.randn(3, 2), np.random.randn(10, 2)
    g = calc.grad_linear_least_squares(X, W, Y)
    g_num = calc.numerical_gradient(lambda w: float(np.sum((X @ w - Y) ** 2)), W)
    assert calc.relative_error(g, g_num) < 1e-6
    Wt = torch.tensor(W, requires_grad=True)
    ((torch.tensor(X) @ Wt - torch.tensor(Y)) ** 2).sum().backward()
    assert np.allclose(g, Wt.grad.numpy())


def test_jacobian_and_hessian():
    A = np.random.randn(3, 4)
    x = np.random.randn(4)
    J = calc.numerical_jacobian(lambda z: A @ z, x)
    assert np.allclose(J, A, atol=1e-6)
    M = np.random.randn(4, 4)
    S = M @ M.T
    H = calc.numerical_hessian(lambda z: 0.5 * float(z @ S @ z), x)
    assert np.allclose(H, S, atol=1e-4)


def test_softmax_backward_matches_torch():
    S = np.random.randn(3, 5)
    dA = np.random.randn(3, 5)
    St = torch.tensor(S, requires_grad=True)
    A_t = torch.softmax(St, dim=-1)
    A_t.backward(torch.tensor(dA))
    dS = calc.softmax_backward(dA, A_t.detach().numpy())
    assert np.allclose(dS, St.grad.numpy(), atol=1e-10)


def test_attention_backward_matches_torch_autograd():
    T, d_k, d_v = 5, 4, 3
    Q, K, V = (np.random.randn(T, d) for d in (d_k, d_k, d_v))
    dY = np.random.randn(T, d_v)
    Y, cache = calc.attention_forward(Q, K, V)
    dQ, dK, dV = calc.attention_backward(dY, cache)

    Qt, Kt, Vt = (torch.tensor(a, requires_grad=True) for a in (Q, K, V))
    Yt = torch.softmax(Qt @ Kt.T / np.sqrt(d_k), dim=-1) @ Vt
    Yt.backward(torch.tensor(dY))
    assert np.allclose(Y, Yt.detach().numpy(), atol=1e-10)
    assert np.allclose(dQ, Qt.grad.numpy(), atol=1e-8)
    assert np.allclose(dK, Kt.grad.numpy(), atol=1e-8)
    assert np.allclose(dV, Vt.grad.numpy(), atol=1e-8)


def test_attention_backward_vs_finite_difference():
    T, d_k, d_v = 3, 2, 2
    Q, K, V = (np.random.randn(T, d) for d in (d_k, d_k, d_v))
    dY = np.random.randn(T, d_v)
    _, cache = calc.attention_forward(Q, K, V)
    dQ, _, _ = calc.attention_backward(dY, cache)
    loss = lambda q: float(np.sum(calc.attention_forward(q, K, V)[0] * dY))  # noqa: E731
    assert calc.relative_error(dQ, calc.numerical_gradient(loss, Q)) < 1e-5


def test_taylor_second_order_is_exact_for_quadratics():
    S = np.array([[2.0, 0.5], [0.5, 1.0]])
    f = lambda z: float(0.5 * z @ S @ z + z.sum())  # noqa: E731
    x0, x = np.zeros(2), np.array([0.3, -0.7])
    assert np.isclose(calc.taylor_second_order(f, x0, x), f(x), atol=1e-5)
