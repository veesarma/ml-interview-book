"""Scaled dot-product attention: PyTorch vs F.scaled_dot_product_attention; NumPy backward vs autograd."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from mlbook.transformer.attention import attention_backward, attention_flops, attention_forward, scaled_dot_product_attention, softmax_rows
from mlbook.transformer.masks import causal_mask


def test_scaled_dot_product_attention_matches_torch_reference():
    B, H, T, d = 2, 3, 5, 8
    q, k, v = (torch.randn(B, H, T, d) for _ in range(3))
    out, attn = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(attn.sum(-1), torch.ones(B, H, T))


def test_scaled_dot_product_attention_causal_matches_torch_is_causal():
    B, H, T, d = 2, 2, 6, 4
    q, k, v = (torch.randn(B, H, T, d) for _ in range(3))
    out, attn = scaled_dot_product_attention(q, k, v, mask=causal_mask(T))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
    assert torch.all(attn.triu(1) == 0)  # nothing above the diagonal


def test_scaled_dot_product_attention_scale_is_sqrt_dk():
    q, k, v = (torch.randn(1, 1, 3, 16) for _ in range(3))
    _, attn = scaled_dot_product_attention(q, k, v)
    manual = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(16), dim=-1)
    torch.testing.assert_close(attn, manual)


def test_softmax_rows_is_stable_and_normalised():
    S = np.array([[1000.0, 1000.0], [0.0, -1e3]])
    A = softmax_rows(S)
    np.testing.assert_allclose(A.sum(-1), 1.0)
    np.testing.assert_allclose(A[0], [0.5, 0.5])


def test_attention_forward_numpy_matches_torch():
    T_q, T_k, d_k, d_v = 4, 6, 8, 5
    Q, K, V = np.random.randn(T_q, d_k), np.random.randn(T_k, d_k), np.random.randn(T_k, d_v)
    Y, _ = attention_forward(Q, K, V)
    ref = F.scaled_dot_product_attention(torch.tensor(Q)[None, None], torch.tensor(K)[None, None], torch.tensor(V)[None, None])[0, 0]
    np.testing.assert_allclose(Y, ref.numpy(), atol=1e-10)


def test_attention_backward_numpy_matches_autograd():
    T_q, T_k, d_k, d_v = 4, 6, 8, 5
    Q, K, V = np.random.randn(T_q, d_k), np.random.randn(T_k, d_k), np.random.randn(T_k, d_v)
    mask = np.tril(np.ones((T_q, T_k), dtype=bool), k=2)
    dY = np.random.randn(T_q, d_v)
    Y, cache = attention_forward(Q, K, V, mask)
    dQ, dK, dV = attention_backward(dY, cache)

    tq, tk, tv = (torch.tensor(x, requires_grad=True) for x in (Q, K, V))
    S = tq @ tk.T / math.sqrt(d_k)
    S = S.masked_fill(~torch.tensor(mask), -1e30)
    Yt = F.softmax(S, dim=-1) @ tv
    (Yt * torch.tensor(dY)).sum().backward()
    np.testing.assert_allclose(Y, Yt.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dQ, tq.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(dK, tk.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(dV, tv.grad.numpy(), atol=1e-8)


def test_attention_flops_formula():
    f = attention_flops(B=1, T=1024, d_model=768)
    assert f["projections"] == 4 * 2 * 1024 * 768 * 768
    assert f["qk_t"] == f["av"] == 2 * 1024 * 1024 * 768
    assert f["total"] == f["projections"] + f["qk_t"] + f["av"]
