import torch

from mlbook.systems import tensor_parallel_toy as tp


def test_column_and_row_shards_reassemble():
    W = torch.randn(8, 12)
    assert torch.equal(torch.cat(tp.column_shards(W, 4), dim=1), W)
    assert torch.equal(torch.cat(tp.row_shards(W, 4), dim=0), W)


def test_column_parallel_forward_matches_unsharded():
    X, W = torch.randn(5, 8), torch.randn(8, 12)
    Y = torch.cat(tp.column_parallel_forward(X, tp.column_shards(W, 3)), dim=1)  # (5, 12)
    assert torch.allclose(Y, X @ W, atol=1e-5)


def test_row_parallel_forward_all_reduce_matches_unsharded():
    X, W = torch.randn(5, 12), torch.randn(12, 8)
    Y = tp.row_parallel_forward(list(torch.chunk(X, 4, dim=1)), tp.row_shards(W, 4))  # (5, 8)
    assert torch.allclose(Y, X @ W, atol=1e-5)


def test_tensor_parallel_mlp_equals_reference():
    X, A, B = torch.randn(6, 16), torch.randn(16, 64), torch.randn(64, 16)
    for t in (1, 2, 4, 8):
        assert torch.allclose(tp.tensor_parallel_mlp(X, A, B, t), tp.reference_mlp(X, A, B), atol=1e-4)


def test_tensor_parallel_attention_equals_reference():
    T, d, H = 7, 32, 8
    X = torch.randn(T, d)
    Ws = [torch.randn(d, d) / d**0.5 for _ in range(4)]
    ref = tp.reference_attention(X, *Ws, n_heads=H)
    for t in (2, 4, 8):
        assert torch.allclose(tp.tensor_parallel_attention(X, *Ws, n_heads=H, tp=t), ref, atol=1e-4)


def test_simulated_all_reduce_is_sum():
    parts = [torch.ones(2, 2) * i for i in range(4)]
    assert torch.equal(tp.simulated_all_reduce(parts), torch.full((2, 2), 6.0))
