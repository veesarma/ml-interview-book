"""Finite-difference checks for every NumPy layer and activation (Part III, ch. 1-2)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mlbook.nn.layers import GELU, Linear, ReLU, Sigmoid, SiLU, Softmax, Tanh, numerical_gradient, rel_error


def _check_input_grad(layer, x, tol=1e-6):
    """Compare layer.backward against finite differences of loss = sum(out * R)."""
    r = np.random.randn(*layer.forward(x).shape)  # random upstream weights
    layer.forward(x)
    dx = layer.backward(r)
    num = numerical_gradient(lambda a: float((layer.forward(a) * r).sum()), x.copy())
    assert rel_error(dx, num) < tol


@pytest.mark.parametrize("layer_cls", [ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax])
def test_activation_backward_matches_finite_difference(layer_cls):
    x = np.random.randn(4, 5) + 0.05  # avoid the ReLU kink at exactly 0
    _check_input_grad(layer_cls(), x)


def test_linear_backward_all_grads():
    lin = Linear(3, 2)
    x = np.random.randn(5, 3)
    r = np.random.randn(5, 2)
    lin.forward(x)
    dx = lin.backward(r)
    assert dx.shape == (5, 3) and lin.dW.shape == (3, 2) and lin.db.shape == (2,)
    num_dx = numerical_gradient(lambda a: float((lin.forward(a) * r).sum()), x.copy())
    assert rel_error(dx, num_dx) < 1e-6

    def loss_w(w):
        old = lin.W.copy()
        lin.W = w
        val = float((lin.forward(x) * r).sum())
        lin.W = old
        return val

    def loss_b(b):
        old = lin.b.copy()
        lin.b = b
        val = float((lin.forward(x) * r).sum())
        lin.b = old
        return val

    assert rel_error(lin.dW, numerical_gradient(loss_w, lin.W.copy())) < 1e-6
    assert rel_error(lin.db, numerical_gradient(loss_b, lin.b.copy())) < 1e-6
    # closed forms from the chapter
    np.testing.assert_allclose(lin.dW, x.T @ r)
    np.testing.assert_allclose(lin.db, r.sum(0))
    np.testing.assert_allclose(dx, r @ lin.W.T)


def test_activations_match_torch():
    x = np.random.randn(6, 7)
    xt = torch.tensor(x)
    np.testing.assert_allclose(GELU().forward(x), F.gelu(xt).numpy(), atol=1e-6)
    np.testing.assert_allclose(SiLU().forward(x), F.silu(xt).numpy(), atol=1e-12)
    np.testing.assert_allclose(Sigmoid().forward(x), torch.sigmoid(xt).numpy(), atol=1e-12)
    np.testing.assert_allclose(Softmax().forward(x), torch.softmax(xt, -1).numpy(), atol=1e-12)


def test_sigmoid_stable_for_large_inputs():
    s = Sigmoid().forward(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(s))


def test_dead_relu_has_zero_gradient():
    relu = ReLU()
    relu.forward(np.array([[-2.0, -1.0, 3.0]]))
    dx = relu.backward(np.ones((1, 3)))
    np.testing.assert_array_equal(dx, [[0.0, 0.0, 1.0]])
