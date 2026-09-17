import torch

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers

from mlbook.multimodal.projectors import (  # noqa: E402
    GatedCrossAttentionAdapter,
    LinearProjector,
    MLPProjector,
    PerceiverResampler,
    QFormer,
)
from mlbook.multimodal.token_compression import (  # noqa: E402
    PixelShuffleProjector,
    anyres_tiles,
    average_pool_tokens,
    pixel_shuffle_merge,
    prune_tokens_by_score,
    visual_token_count,
)

FEATS = torch.randn(2, 9, 12)  # (B, N_v, d_v)


def test_linear_projector():
    p = LinearProjector(12, 20)
    assert p(FEATS).shape == (2, 9, 20)
    assert torch.allclose(p(FEATS), FEATS @ p.proj.weight.T + p.proj.bias)


def test_mlp_projector():
    assert MLPProjector(12, 20)(FEATS).shape == (2, 9, 20)


def test_perceiver_resampler_fixed_output_length():
    r = PerceiverResampler(12, 20, n_queries=4, n_heads=2)
    assert r(FEATS).shape == (2, 4, 20)
    assert r(torch.randn(2, 50, 12)).shape == (2, 4, 20)  # independent of N_v


def test_qformer_fixed_output_length():
    q = QFormer(12, 20, n_queries=4, n_heads=2)
    assert q(FEATS).shape == (2, 4, 20)
    assert q(torch.randn(2, 33, 12)).shape == (2, 4, 20)


def test_gated_cross_attention_is_identity_at_init_then_changes():
    text = torch.randn(2, 7, 20)
    ad = GatedCrossAttentionAdapter(20, 12, n_heads=2)
    assert torch.allclose(ad(text, FEATS), text)  # tanh(0) = 0
    with torch.no_grad():
        ad.gate_attn.fill_(1.0)
    assert ad(text, FEATS).shape == (2, 7, 20)
    assert not torch.allclose(ad(text, FEATS), text)


def test_average_pool_tokens():
    grid = torch.randn(2, 6, 6, 8)
    out = average_pool_tokens(grid, 2)
    assert out.shape == (2, 9, 8)
    assert torch.allclose(out[0, 0], grid[0, :2, :2].mean(dim=(0, 1)))


def test_pixel_shuffle_merge_keeps_every_value():
    grid = torch.randn(2, 6, 6, 8)
    ps = pixel_shuffle_merge(grid, 3)
    assert ps.shape == (2, 4, 72)
    assert torch.equal(ps[0, 0], grid[0, :3, :3].reshape(-1))
    assert PixelShuffleProjector(8, 16, 2)(grid).shape == (2, 9, 16)


def test_prune_tokens_by_score():
    tokens = torch.randn(2, 36, 8)
    scores = torch.randn(2, 36)
    kept, idx = prune_tokens_by_score(tokens, scores, keep=10)
    assert kept.shape == (2, 10, 8) and torch.all(idx[:, 1:] > idx[:, :-1])
    assert torch.equal(kept[0, 0], tokens[0, idx[0, 0]])


def test_anyres_tiles_and_token_count():
    assert anyres_tiles(torch.randn(1, 3, 8, 12), 4).shape == (6, 3, 4, 4)
    assert visual_token_count(336, 336, 14) == 576
    assert visual_token_count(448, 448, 14, merge=2) == 256
    assert visual_token_count(448, 448, 14, n_queries=64) == 64
