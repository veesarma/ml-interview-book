"""End-to-end MLP: every parameter gradient vs finite differences; > 95% on two moons."""
import numpy as np

from mlbook.nn.layers import numerical_gradient, rel_error
from mlbook.nn.losses import CrossEntropyLoss
from mlbook.nn.mlp import MLP, accuracy, make_two_moons, train_classifier


def test_every_parameter_gradient_matches_finite_difference():
    model = MLP([3, 5, 4, 2], activation="tanh", seed=1)  # tanh: smooth, so FD is clean
    x = np.random.randn(6, 3)
    y = np.array([0, 1, 1, 0, 1, 0])
    loss_fn = CrossEntropyLoss()
    loss_fn.forward(model.forward(x), y)
    dx = model.backward(loss_fn.backward())
    analytic = [g.copy() for g in model.grads()]

    def loss_at(param, value):
        old = param.copy()
        param[...] = value
        val = CrossEntropyLoss().forward(model.forward(x), y)
        param[...] = old
        return val

    for p, g in zip(model.params(), analytic):
        num = numerical_gradient(lambda v, p=p: loss_at(p, v), p.copy())
        assert rel_error(g, num) < 1e-5
    num_dx = numerical_gradient(lambda v: CrossEntropyLoss().forward(model.forward(v), y), x.copy())
    assert rel_error(dx, num_dx) < 1e-5


def test_mlp_reaches_95_percent_on_two_moons():
    x, y = make_two_moons(n=400, noise=0.1, seed=0)
    model = MLP([2, 32, 32, 2], activation="relu", seed=0)
    history = train_classifier(model, x, y, epochs=150, lr=0.1, batch_size=32)
    assert history[-1] < history[0]
    assert accuracy(model, x, y) > 0.95


def test_shapes_and_relu_variant_gradient():
    model = MLP([4, 6, 3], activation="relu", seed=3)
    x = np.random.randn(7, 4) + 0.1
    logits = model.forward(x)
    assert logits.shape == (7, 3)
    assert model.backward(np.ones((7, 3))).shape == (7, 4)
