"""The NumPy autograd engine vs torch.autograd on random graphs."""
import numpy as np
import torch

from mlbook.nn.autograd import Tensor, cross_entropy, unbroadcast


def _pair(shape):
    a = np.random.randn(*shape)
    return Tensor(a, requires_grad=True), torch.tensor(a, requires_grad=True)


def test_unbroadcast_sums_over_broadcast_axes():
    g = np.ones((4, 3))
    assert unbroadcast(g, (3,)).shape == (3,)
    np.testing.assert_array_equal(unbroadcast(g, (3,)), [4, 4, 4])
    assert unbroadcast(g, (1, 3)).shape == (1, 3)
    assert unbroadcast(g, (4, 1)).shape == (4, 1)
    np.testing.assert_array_equal(unbroadcast(g, (4, 1)), np.full((4, 1), 3))


def test_two_layer_mlp_matches_torch():
    x_n, x_t = _pair((5, 3))
    w1_n, w1_t = _pair((3, 4))
    b1_n, b1_t = _pair((4,))
    w2_n, w2_t = _pair((4, 2))
    b2_n, b2_t = _pair((2,))
    y = np.array([0, 1, 1, 0, 1])

    h_n = (x_n @ w1_n + b1_n).relu()
    loss_n = cross_entropy(h_n @ w2_n + b2_n, y)
    loss_n.backward()

    h_t = torch.relu(x_t @ w1_t + b1_t)
    loss_t = torch.nn.functional.cross_entropy(h_t @ w2_t + b2_t, torch.tensor(y))
    loss_t.backward()

    assert abs(loss_n.data - loss_t.item()) < 1e-10
    for n, t in [(x_n, x_t), (w1_n, w1_t), (b1_n, b1_t), (w2_n, w2_t), (b2_n, b2_t)]:
        np.testing.assert_allclose(n.grad, t.grad.numpy(), atol=1e-10)


def test_random_elementwise_graph_matches_torch():
    a_n, a_t = _pair((3, 4))
    b_n, b_t = _pair((4,))
    c_n, c_t = _pair((3, 1))
    out_n = ((a_n * b_n + c_n).tanh().exp() / (a_n ** 2 + 1.0)).sigmoid()
    out_n = (out_n.log() * a_n - c_n).sum()
    out_n.backward()
    out_t = ((a_t * b_t + c_t).tanh().exp() / (a_t ** 2 + 1.0)).sigmoid()
    out_t = (out_t.log() * a_t - c_t).sum()
    out_t.backward()
    assert abs(out_n.data - out_t.item()) < 1e-10
    for n, t in [(a_n, a_t), (b_n, b_t), (c_n, c_t)]:
        np.testing.assert_allclose(n.grad, t.grad.numpy(), atol=1e-9)


def test_softmax_logsoftmax_reshape_transpose_mean_match_torch():
    a_n, a_t = _pair((2, 3, 4))
    w_n, w_t = _pair((4, 3))
    s_n = (a_n @ w_n).softmax(axis=-1)  # (2, 3, 3)
    l_n = a_n.log_softmax(axis=1).transpose(0, 2, 1).reshape(8, 3).mean(axis=0)  # (3,)
    out_n = (s_n * s_n).sum() + (l_n * l_n).sum()
    out_n.backward()
    s_t = torch.softmax(a_t @ w_t, dim=-1)
    l_t = torch.log_softmax(a_t, dim=1).permute(0, 2, 1).reshape(8, 3).mean(dim=0)
    out_t = (s_t * s_t).sum() + (l_t * l_t).sum()
    out_t.backward()
    assert abs(out_n.data - out_t.item()) < 1e-10
    np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), atol=1e-9)
    np.testing.assert_allclose(w_n.grad, w_t.grad.numpy(), atol=1e-9)


def test_batched_matmul_with_broadcast_matches_torch():
    a_n, a_t = _pair((4, 2, 3))
    b_n, b_t = _pair((3, 5))  # broadcast over the batch axis
    (a_n @ b_n).sum().backward()
    (a_t @ b_t).sum().backward()
    np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(b_n.grad, b_t.grad.numpy(), atol=1e-10)


def test_gradient_accumulates_when_tensor_used_twice():
    a = Tensor([2.0], requires_grad=True)
    (a * a + a).backward()  # d/da (a^2 + a) = 2a + 1 = 5
    np.testing.assert_allclose(a.grad, [5.0])


def test_diamond_graph_topological_order():
    x = Tensor([1.5], requires_grad=True)
    y = x * 2.0
    z = x * 3.0
    (y * z).sum().backward()  # 6 x^2 -> 12 x = 18
    np.testing.assert_allclose(x.grad, [18.0])


def _grad_vs_torch(build_np, build_t, shape=(3, 4)):
    a_n, a_t = _pair(shape)
    r = np.random.randn(*build_np(a_n).shape)
    (build_np(a_n) * Tensor(r)).sum().backward()
    (build_t(a_t) * torch.tensor(r)).sum().backward()
    np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), atol=1e-9)


def test_op_add_and_mul_broadcast():
    _grad_vs_torch(lambda a: a + a * 3.0 + Tensor(np.ones(4)), lambda a: a + a * 3.0 + torch.ones(4, dtype=torch.float64))


def test_op_pow_and_div():
    _grad_vs_torch(lambda a: (a ** 3) / (a ** 2 + 1.0), lambda a: (a ** 3) / (a ** 2 + 1.0))


def test_op_exp_log():
    _grad_vs_torch(lambda a: (a.exp() + 1.0).log(), lambda a: (a.exp() + 1.0).log())


def test_op_relu():
    _grad_vs_torch(lambda a: a.relu(), lambda a: torch.relu(a))


def test_op_sigmoid_tanh():
    _grad_vs_torch(lambda a: a.sigmoid() * a.tanh(), lambda a: torch.sigmoid(a) * torch.tanh(a))


def test_op_sum_mean_axes():
    _grad_vs_torch(lambda a: a.sum(axis=1, keepdims=True) * a.mean(axis=0), lambda a: a.sum(dim=1, keepdim=True) * a.mean(dim=0))


def test_op_softmax():
    _grad_vs_torch(lambda a: a.softmax(axis=-1), lambda a: torch.softmax(a, dim=-1))


def test_op_log_softmax():
    _grad_vs_torch(lambda a: a.log_softmax(axis=0), lambda a: torch.log_softmax(a, dim=0))


def test_op_reshape_transpose():
    _grad_vs_torch(lambda a: a.reshape(2, 6).transpose(1, 0) * Tensor(np.arange(12.0).reshape(6, 2)),
                   lambda a: a.reshape(2, 6).permute(1, 0) * torch.arange(12.0, dtype=torch.float64).reshape(6, 2))


def test_op_matmul_2d():
    a_n, a_t = _pair((3, 4))
    b_n, b_t = _pair((4, 2))
    (a_n @ b_n).sum().backward()
    (a_t @ b_t).sum().backward()
    np.testing.assert_allclose(a_n.grad, a_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(b_n.grad, b_t.grad.numpy(), atol=1e-10)


def test_cross_entropy_helper_matches_torch():
    z_n, z_t = _pair((6, 3))
    y = np.array([0, 2, 1, 1, 0, 2])
    cross_entropy(z_n, y).backward()
    torch.nn.functional.cross_entropy(z_t, torch.tensor(y)).backward()
    np.testing.assert_allclose(z_n.grad, z_t.grad.numpy(), atol=1e-10)
