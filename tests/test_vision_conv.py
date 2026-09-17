import numpy as np
import torch
import torch.nn.functional as F

from mlbook.vision import conv as C


def test_output_size_and_receptive_field():
    assert C.conv_output_size(224, 7, stride=2, pad=3) == 112
    assert C.conv_output_size(10, 3, stride=1, pad=0, dilation=2) == 6
    # VGG-style: three 3x3 stride-1 convs → RF 7; add a stride-2 3x3 → RF 9, jump 2
    assert C.receptive_field([(3, 1, 1)] * 3) == (7, 1)
    assert C.receptive_field([(3, 1, 1)] * 3 + [(3, 2, 1)]) == (9, 2)
    assert C.receptive_field([(7, 2, 1), (3, 2, 1)]) == (11, 4)


def test_flops_depthwise_separable_ratio():
    sep, std = C.depthwise_separable_flops(c_in=256, c_out=256, k=3, h_out=14, w_out=14)
    assert abs(sep / std - (1 / 256 + 1 / 9)) < 1e-9
    macs, params = C.conv_flops_and_params(64, 128, 3, 56, 56)
    assert params == 128 * 64 * 9 + 128 and macs == 56 * 56 * 128 * 64 * 9


def test_naive_and_im2col_match_torch_forward():
    x = np.random.randn(2, 3, 9, 8)
    w = np.random.randn(4, 3, 3, 3)
    b = np.random.randn(4)
    for stride, pad, dil in [(1, 0, 1), (2, 1, 1), (1, 2, 2)]:
        ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=stride, padding=pad, dilation=dil).numpy()
        out, _ = C.conv2d_im2col(x, w, b, stride=stride, pad=pad, dilation=dil)
        assert np.allclose(out, ref, atol=1e-10)
        naive = C.conv2d_naive(x, w, stride=stride, pad=pad, dilation=dil) + b[None, :, None, None]
        assert np.allclose(naive, ref, atol=1e-10)


def test_grouped_and_depthwise_forward():
    x = np.random.randn(1, 6, 7, 7)
    w = np.random.randn(6, 1, 3, 3)  # depthwise: groups = C_in = C_out
    out, _ = C.conv2d_im2col(x, w, stride=1, pad=1, groups=6)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), padding=1, groups=6).numpy()
    assert np.allclose(out, ref)


def test_backward_matches_torch_autograd():
    x = np.random.randn(2, 4, 8, 7)
    w = np.random.randn(6, 2, 3, 3)
    b = np.random.randn(6)
    out, cache = C.conv2d_im2col(x, w, b, stride=2, pad=1, groups=2)
    dout = np.random.randn(*out.shape)
    dx, dw, db = C.conv2d_backward(dout, w, cache)
    xt, wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
    F.conv2d(xt, wt, bt, stride=2, padding=1, groups=2).backward(torch.tensor(dout))
    assert np.allclose(dx, xt.grad.numpy(), atol=1e-10)
    assert np.allclose(dw, wt.grad.numpy(), atol=1e-10)
    assert np.allclose(db, bt.grad.numpy(), atol=1e-10)


def test_backward_finite_difference():
    x = np.random.randn(1, 2, 5, 5)
    w = np.random.randn(3, 2, 3, 3)
    out, cache = C.conv2d_im2col(x, w, stride=1, pad=1)
    dout = np.random.randn(*out.shape)
    dx, _, _ = C.conv2d_backward(dout, w, cache)
    eps = 1e-6
    for _ in range(5):
        idx = tuple(np.random.randint(s) for s in x.shape)
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num = (np.sum(C.conv2d_im2col(xp, w, pad=1)[0] * dout) - np.sum(C.conv2d_im2col(xm, w, pad=1)[0] * dout)) / (2 * eps)
        assert abs(num - dx[idx]) < 1e-5


def test_transposed_conv_matches_torch():
    x = np.random.randn(2, 3, 4, 5)
    w = np.random.randn(3, 5, 3, 3)
    for stride, pad in [(1, 0), (2, 1), (2, 0)]:
        ours = C.conv_transpose2d(x, w, stride=stride, pad=pad)
        ref = F.conv_transpose2d(torch.tensor(x), torch.tensor(w), stride=stride, padding=pad).numpy()
        assert ours.shape == ref.shape and np.allclose(ours, ref, atol=1e-10)


def test_im2col_col2im_are_adjoint():
    # <im2col(x), c> == <x, col2im(c)> for random x, c (the defining property of an adjoint)
    x = np.random.randn(1, 2, 6, 6)
    cols = C.im2col(x, 3, 3, stride=2, pad=1)
    c = np.random.randn(*cols.shape)
    lhs = np.sum(cols * c)
    rhs = np.sum(x * C.col2im(c, x.shape, 3, 3, 2, 1, 1))
    assert abs(lhs - rhs) < 1e-10
