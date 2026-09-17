"""NumPy norms vs torch.nn.functional (forward + backward); torch modules vs F.*."""
import numpy as np
import torch
import torch.nn.functional as F

from mlbook.nn import normalization as npn
from mlbook.nn import normalization_torch as tn


def _torch_grads(fn, *arrays):
    ts = [torch.tensor(a, requires_grad=True) for a in arrays]
    out = fn(*ts)
    r = torch.tensor(np.random.randn(*out.shape))
    (out * r).sum().backward()
    return out.detach().numpy(), r.numpy(), [t.grad.numpy() for t in ts]


def test_batchnorm1d_train_forward_backward_vs_torch():
    x = np.random.randn(8, 5) * 2 + 1
    bn = npn.BatchNorm1d(5)
    bn.gamma = np.random.randn(5)
    bn.beta = np.random.randn(5)
    out_t, r, (dx_t, dg_t, db_t) = _torch_grads(
        lambda xt, g, b: F.batch_norm(xt, None, None, g, b, training=True, eps=1e-5), x, bn.gamma, bn.beta
    )
    out = bn.forward(x)
    np.testing.assert_allclose(out, out_t, atol=1e-10)
    dx = bn.backward(r)
    np.testing.assert_allclose(dx, dx_t, atol=1e-10)
    np.testing.assert_allclose(bn.dgamma, dg_t, atol=1e-10)
    np.testing.assert_allclose(bn.dbeta, db_t, atol=1e-10)


def test_batchnorm_running_stats_and_eval_match_torch():
    x = np.random.randn(16, 3) * 3 - 2
    bn = npn.BatchNorm1d(3, momentum=0.1)
    ref = torch.nn.BatchNorm1d(3, momentum=0.1).double()
    for _ in range(3):
        bn.forward(x)
        ref(torch.tensor(x))
    np.testing.assert_allclose(bn.running_mean, ref.running_mean.numpy(), atol=1e-10)
    np.testing.assert_allclose(bn.running_var, ref.running_var.numpy(), atol=1e-10)
    bn.training = False
    ref.eval()
    x_eval = np.random.randn(4, 3)
    np.testing.assert_allclose(bn.forward(x_eval), ref(torch.tensor(x_eval)).detach().numpy(), atol=1e-10)


def test_batchnorm2d_vs_torch():
    x = np.random.randn(4, 3, 5, 5)
    bn = npn.BatchNorm2d(3)
    out_t, r, (dx_t, dg_t, db_t) = _torch_grads(
        lambda xt, g, b: F.batch_norm(xt, None, None, g, b, training=True, eps=1e-5), x, bn.bn.gamma, bn.bn.beta
    )
    np.testing.assert_allclose(bn.forward(x), out_t, atol=1e-10)
    np.testing.assert_allclose(bn.backward(r), dx_t, atol=1e-10)
    np.testing.assert_allclose(bn.bn.dgamma, dg_t, atol=1e-10)


def test_layernorm_vs_torch():
    x = np.random.randn(2, 7, 6)
    ln = npn.LayerNorm(6)
    ln.gamma, ln.beta = np.random.randn(6), np.random.randn(6)
    out_t, r, (dx_t, dg_t, db_t) = _torch_grads(lambda xt, g, b: F.layer_norm(xt, (6,), g, b, eps=1e-5), x, ln.gamma, ln.beta)
    np.testing.assert_allclose(ln.forward(x), out_t, atol=1e-10)
    np.testing.assert_allclose(ln.backward(r), dx_t, atol=1e-10)
    np.testing.assert_allclose(ln.dgamma, dg_t, atol=1e-10)
    np.testing.assert_allclose(ln.dbeta, db_t, atol=1e-10)


def test_rmsnorm_vs_torch():
    x = np.random.randn(3, 4, 8)
    rms = npn.RMSNorm(8, eps=1e-6)
    rms.g = np.random.randn(8)
    out_t, r, (dx_t, dg_t) = _torch_grads(lambda xt, g: F.rms_norm(xt, (8,), g, eps=1e-6), x, rms.g)
    np.testing.assert_allclose(rms.forward(x), out_t, atol=1e-10)
    np.testing.assert_allclose(rms.backward(r), dx_t, atol=1e-10)
    np.testing.assert_allclose(rms.dg, dg_t, atol=1e-10)


def test_groupnorm_vs_torch():
    x = np.random.randn(2, 6, 4, 4)
    gn = npn.GroupNorm(3, 6)
    gn.gamma, gn.beta = np.random.randn(6), np.random.randn(6)
    out_t, r, (dx_t, dg_t, db_t) = _torch_grads(lambda xt, g, b: F.group_norm(xt, 3, g, b, eps=1e-5), x, gn.gamma, gn.beta)
    np.testing.assert_allclose(gn.forward(x), out_t, atol=1e-10)
    np.testing.assert_allclose(gn.backward(r), dx_t, atol=1e-10)
    np.testing.assert_allclose(gn.dgamma, dg_t, atol=1e-10)
    np.testing.assert_allclose(gn.dbeta, db_t, atol=1e-10)


def test_torch_modules_match_functional():
    x = torch.randn(10, 6, dtype=torch.float64)
    bn = tn.BatchNorm1d(6).double()
    torch.testing.assert_close(bn(x), F.batch_norm(x, None, None, bn.gamma, bn.beta, training=True))
    bn.eval()
    torch.testing.assert_close(bn(x), F.batch_norm(x, bn.running_mean, bn.running_var, bn.gamma, bn.beta, training=False))
    ln = tn.LayerNorm(6).double()
    torch.testing.assert_close(ln(x), F.layer_norm(x, (6,), ln.gamma, ln.beta))
    rms = tn.RMSNorm(6).double()
    torch.testing.assert_close(rms(x), F.rms_norm(x, (6,), rms.g, eps=1e-6))
