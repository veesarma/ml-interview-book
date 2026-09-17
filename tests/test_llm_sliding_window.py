"""Tests for sparse-attention masks and the rolling KV buffer (Part VI, chapter 3)."""

import torch

from mlbook.llm.sliding_window import (
    RollingKVCache,
    attention_sink_mask,
    block_sparse_mask,
    masked_attention,
    sliding_window_mask,
)


def test_sliding_window_mask_values():
    m = sliding_window_mask(5, window=2)
    expected = torch.tensor(
        [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(m, expected)
    assert torch.equal(sliding_window_mask(4, window=100), torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_attention_sink_mask_keeps_first_tokens():
    m = attention_sink_mask(6, window=2, n_sink=1)
    assert m[5, 0] and m[5, 4] and m[5, 5] and not m[5, 1]


def test_block_sparse_mask_local_and_global():
    m = block_sparse_mask(8, block=2, global_tokens=1)
    assert m[7, 0]  # global token visible to all
    assert m[7, 6] and m[7, 4] and not m[7, 3]  # own block + previous block only
    assert not m[0, 1]  # still causal


def test_rolling_cache_matches_windowed_attention_for_last_token():
    torch.manual_seed(0)
    T, W, H_kv, d = 10, 4, 2, 8
    q = torch.randn(1, H_kv, T, d)
    k = torch.randn(1, H_kv, T, d)
    v = torch.randn(1, H_kv, T, d)
    full = masked_attention(q, k, v, sliding_window_mask(T, W))  # (1, H_kv, T, d)
    cache = RollingKVCache(W, H_kv, d)
    for t in range(T):
        cache.append(k[:, :, t : t + 1], v[:, :, t : t + 1])
    k_w, v_w = cache.window_kv()  # last W tokens in chronological order
    assert k_w.shape == (1, H_kv, W, d)
    last = masked_attention(q[:, :, T - 1 : T], k_w, v_w, torch.ones(1, W, dtype=torch.bool))
    assert torch.allclose(last[:, :, 0], full[:, :, T - 1], atol=1e-6)
