import numpy as np
import torch

from mlbook.math import tensor_ops as to


def test_split_heads():
    x = np.random.randn(2, 5, 12)
    y = to.split_heads(x, H=3)
    assert y.shape == (2, 3, 5, 4)
    # head h of token t is the h-th contiguous chunk of that token's vector
    assert np.allclose(y[1, 2, 4], x[1, 4, 8:12])
    ref = torch.tensor(x).view(2, 5, 3, 4).transpose(1, 2).numpy()
    assert np.allclose(y, ref)


def test_merge_heads():
    x = np.random.randn(2, 5, 12)
    assert np.allclose(to.merge_heads(to.split_heads(x, 3)), x)
    y = np.random.randn(2, 3, 5, 4)
    ref = torch.tensor(y).transpose(1, 2).contiguous().view(2, 5, 12).numpy()
    assert np.allclose(to.merge_heads(y), ref)


def test_batched_outer():
    u, v = np.random.randn(4, 3), np.random.randn(4, 5)
    out = to.batched_outer(u, v)
    assert out.shape == (4, 3, 5)
    assert np.allclose(out[2], np.outer(u[2], v[2]))
    assert np.allclose(out, torch.einsum("bm,bn->bmn", torch.tensor(u), torch.tensor(v)).numpy())


def test_causal_mask():
    m = to.causal_mask(4)
    assert m.shape == (1, 1, 4, 4)
    assert np.isneginf(m[0, 0, 0, 1]) and m[0, 0, 3, 0] == 0.0 and m[0, 0, 2, 2] == 0.0


def test_padding_mask():
    m = to.padding_mask(np.array([2, 4]), T=4)
    assert m.shape == (2, 1, 1, 4)
    assert np.isneginf(m[0, 0, 0, 2]) and m[0, 0, 0, 1] == 0.0 and np.all(m[1] == 0.0)


def test_masked_attention():
    B, H, T, d = 2, 3, 5, 4
    Q, K, V = (np.random.randn(B, H, T, d) for _ in range(3))
    out = to.masked_attention(Q, K, V, to.causal_mask(T))
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(Q), torch.tensor(K), torch.tensor(V), is_causal=True
    ).numpy()
    assert np.allclose(out, ref, atol=1e-10)
    # padding mask + causal mask combine by broadcasting addition
    pad = to.padding_mask(np.array([3, 5]), T)
    out2 = to.masked_attention(Q, K, V, to.causal_mask(T) + pad)
    assert np.allclose(out2[1], ref[1])  # sample 1 has no padding
    assert not np.allclose(out2[0, :, 4], ref[0, :, 4])  # sample 0 query 4 can't see keys 3, 4


def test_gather_token_logprobs():
    lp = np.log(np.random.dirichlet(np.ones(7), size=(2, 3)))  # (2, 3, 7)
    tgt = np.array([[0, 6, 3], [2, 2, 5]])
    out = to.gather_token_logprobs(lp, tgt)
    assert out.shape == (2, 3)
    assert out[1, 2] == lp[1, 2, 5]
    ref = torch.gather(torch.tensor(lp), 2, torch.tensor(tgt)[:, :, None])[:, :, 0].numpy()
    assert np.allclose(out, ref)


def test_one_hot():
    y = np.array([0, 2, 1])
    oh = to.one_hot(y, 3)
    assert np.allclose(oh, np.eye(3)[y])
    assert np.allclose(oh, torch.nn.functional.one_hot(torch.tensor(y), 3).numpy())


def test_cross_entropy_from_logits():
    logits = np.random.randn(6, 4) * 5
    y = np.array([0, 1, 2, 3, 0, 1])
    ours = to.cross_entropy_from_logits(logits, y)
    ref = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(y)).item()
    assert np.isclose(ours, ref)


def test_attention_einsum():
    B, H, T, d = 2, 2, 4, 3
    Q, K, V = (np.random.randn(B, H, T, d) for _ in range(3))
    out = to.attention_einsum(Q, K, V)
    ref = torch.nn.functional.scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V)).numpy()
    assert np.allclose(out, ref, atol=1e-10)
    assert np.allclose(out, to.masked_attention(Q, K, V, np.zeros((1, 1, T, T))))
