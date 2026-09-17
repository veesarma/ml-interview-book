"""One focused test per public symbol in mlbook.math.calculus (select with -k <name>)."""

import numpy as np
import torch

from mlbook.math import calculus as calc


def test_numerical_gradient():
    f = lambda x: float(np.sum(x**3))  # noqa: E731
    x = np.random.randn(5)
    assert np.allclose(calc.numerical_gradient(f, x), 3 * x**2, atol=1e-6)
    W = np.random.randn(3, 2)  # works on matrices too (same-shape output)
    g = calc.numerical_gradient(lambda w: float(np.sum(w**2)), W)
    assert g.shape == W.shape and np.allclose(g, 2 * W, atol=1e-6)


def test_relative_error():
    a = np.array([1.0, 2.0])
    assert calc.relative_error(a, a) == 0.0
    assert np.isclose(calc.relative_error(a, 2 * a), 0.5)


def test_numerical_jacobian():
    A = np.random.randn(3, 4)
    x = np.random.randn(4)
    assert np.allclose(calc.numerical_jacobian(lambda z: A @ z, x), A, atol=1e-6)
    J = calc.numerical_jacobian(lambda z: np.tanh(z), x)  # elementwise -> diagonal
    assert np.allclose(J, np.diag(1 - np.tanh(x) ** 2), atol=1e-6)


def test_numerical_hessian():
    M = np.random.randn(4, 4)
    S = M @ M.T
    x = np.random.randn(4)
    H = calc.numerical_hessian(lambda z: 0.5 * float(z @ S @ z), x)
    assert np.allclose(H, S, atol=1e-4)


def test_grad_quadratic_form():
    A = np.random.randn(4, 4)  # deliberately non-symmetric
    x = np.random.randn(4)
    g = calc.grad_quadratic_form(A, x)
    g_num = calc.numerical_gradient(lambda z: float(z @ A @ z), x)
    assert calc.relative_error(g, g_num) < 1e-6
    S = A + A.T
    assert np.allclose(calc.grad_quadratic_form(S, x), 2 * S @ x)


def test_grad_linear_least_squares():
    X, W, Y = np.random.randn(10, 3), np.random.randn(3, 2), np.random.randn(10, 2)
    g = calc.grad_linear_least_squares(X, W, Y)
    g_num = calc.numerical_gradient(lambda w: float(np.sum((X @ w - Y) ** 2)), W)
    assert calc.relative_error(g, g_num) < 1e-6
    Wt = torch.tensor(W, requires_grad=True)
    ((torch.tensor(X) @ Wt - torch.tensor(Y)) ** 2).sum().backward()
    assert np.allclose(g, Wt.grad.numpy())


def test_softmax_backward():
    S = np.random.randn(3, 5)
    dA = np.random.randn(3, 5)
    St = torch.tensor(S, requires_grad=True)
    A_t = torch.softmax(St, dim=-1)
    A_t.backward(torch.tensor(dA))
    dS = calc.softmax_backward(dA, A_t.detach().numpy())
    assert np.allclose(dS, St.grad.numpy(), atol=1e-10)
    assert np.allclose(dS.sum(axis=-1), 0.0, atol=1e-12)  # rows of dS sum to 0


def test_attention_forward():
    T, d_k, d_v = 5, 4, 3
    Q, K, V = (np.random.randn(T, d) for d in (d_k, d_k, d_v))
    Y, cache = calc.attention_forward(Q, K, V)
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(Q)[None], torch.tensor(K)[None], torch.tensor(V)[None]
    )[0].numpy()
    assert np.allclose(Y, ref, atol=1e-10)
    assert np.allclose(cache["A"].sum(axis=-1), 1.0)


def test_attention_backward():
    T, d_k, d_v = 5, 4, 3
    Q, K, V = (np.random.randn(T, d) for d in (d_k, d_k, d_v))
    dY = np.random.randn(T, d_v)
    _, cache = calc.attention_forward(Q, K, V)
    dQ, dK, dV = calc.attention_backward(dY, cache)
    Qt, Kt, Vt = (torch.tensor(a, requires_grad=True) for a in (Q, K, V))
    Yt = torch.softmax(Qt @ Kt.T / np.sqrt(d_k), dim=-1) @ Vt
    Yt.backward(torch.tensor(dY))
    assert np.allclose(dQ, Qt.grad.numpy(), atol=1e-8)
    assert np.allclose(dK, Kt.grad.numpy(), atol=1e-8)
    assert np.allclose(dV, Vt.grad.numpy(), atol=1e-8)
    # and against finite differences, the way you'd check without autograd
    loss = lambda q: float(np.sum(calc.attention_forward(q, K, V)[0] * dY))  # noqa: E731
    assert calc.relative_error(dQ, calc.numerical_gradient(loss, Q)) < 1e-5


def test_taylor_second_order():
    S = np.array([[2.0, 0.5], [0.5, 1.0]])
    f = lambda z: float(0.5 * z @ S @ z + z.sum())  # noqa: E731
    x0, x = np.zeros(2), np.array([0.3, -0.7])
    assert np.isclose(calc.taylor_second_order(f, x0, x), f(x), atol=1e-5)  # exact for quadratics
