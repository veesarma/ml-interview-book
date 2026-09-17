"""Losses: closed-form gradients, finite differences and torch references."""
import numpy as np
import torch
import torch.nn.functional as F

from mlbook.nn.layers import numerical_gradient, rel_error
from mlbook.nn.losses import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, log_softmax, one_hot, softmax


def test_cross_entropy_gradient_is_p_minus_y_over_n():
    z = np.random.randn(5, 4)
    y = np.array([0, 3, 1, 2, 3])
    loss = CrossEntropyLoss()
    val = loss.forward(z, y)
    grad = loss.backward()
    p = softmax(z)
    np.testing.assert_allclose(grad, (p - one_hot(y, 4)) / 5)
    num = numerical_gradient(lambda a: CrossEntropyLoss().forward(a, y), z.copy())
    assert rel_error(grad, num) < 1e-6
    ref = F.cross_entropy(torch.tensor(z), torch.tensor(y)).item()
    assert abs(val - ref) < 1e-10


def test_log_softmax_stable_for_huge_logits():
    z = np.array([[1000.0, 1000.0, -1000.0]])
    lp = log_softmax(z)
    assert np.all(np.isfinite(lp))
    np.testing.assert_allclose(np.exp(lp).sum(), 1.0)


def test_mse_gradient():
    yhat, y = np.random.randn(3, 2), np.random.randn(3, 2)
    loss = MSELoss()
    loss.forward(yhat, y)
    num = numerical_gradient(lambda a: MSELoss().forward(a, y), yhat.copy())
    assert rel_error(loss.backward(), num) < 1e-6
    ref = F.mse_loss(torch.tensor(yhat), torch.tensor(y)).item()
    assert abs(loss.forward(yhat, y) - ref) < 1e-12


def test_bce_with_logits():
    z = np.random.randn(8)
    y = (np.random.rand(8) > 0.5).astype(float)
    loss = BCEWithLogitsLoss()
    val = loss.forward(z, y)
    ref = F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(y)).item()
    assert abs(val - ref) < 1e-12
    num = numerical_gradient(lambda a: BCEWithLogitsLoss().forward(a, y), z.copy())
    assert rel_error(loss.backward(), num) < 1e-6
