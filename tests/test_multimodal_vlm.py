import torch

from mlbook.multimodal.projectors import GatedCrossAttentionAdapter, LinearProjector, MLPProjector, PerceiverResampler, QFormer
from mlbook.multimodal.token_compression import (
    anyres_tiles,
    average_pool_tokens,
    pixel_shuffle_merge,
    PixelShuffleProjector,
    prune_tokens_by_score,
    visual_token_count,
)
from mlbook.multimodal.vlm import MiniVLM, TinyCausalLM, ToyVisionEncoder, merge_visual_tokens, vlm_lm_loss


def test_projector_shapes():
    feats = torch.randn(2, 9, 12)  # (B, N_v, d_v)
    assert LinearProjector(12, 20)(feats).shape == (2, 9, 20)
    assert MLPProjector(12, 20)(feats).shape == (2, 9, 20)
    assert PerceiverResampler(12, 20, n_queries=4, n_heads=2)(feats).shape == (2, 4, 20)
    assert QFormer(12, 20, n_queries=4, n_heads=2)(feats).shape == (2, 4, 20)
    text = torch.randn(2, 7, 20)
    ad = GatedCrossAttentionAdapter(20, 12, n_heads=2)
    out = ad(text, feats)
    assert out.shape == (2, 7, 20)
    assert torch.allclose(out, text)  # zero-initialised gates: identity at step 0


def test_token_compression():
    grid = torch.randn(2, 6, 6, 8)
    assert average_pool_tokens(grid, 2).shape == (2, 9, 8)
    ps = pixel_shuffle_merge(grid, 3)
    assert ps.shape == (2, 4, 72)
    # pixel shuffle keeps every value: block (0,0) is the concatenation of grid[:, :3, :3]
    assert torch.equal(ps[0, 0], grid[0, :3, :3].reshape(-1))
    assert PixelShuffleProjector(8, 16, 2)(grid).shape == (2, 9, 16)
    kept, idx = prune_tokens_by_score(grid.reshape(2, 36, 8), torch.randn(2, 36), keep=10)
    assert kept.shape == (2, 10, 8) and torch.all(idx[:, 1:] > idx[:, :-1])
    assert anyres_tiles(torch.randn(1, 3, 8, 12), 4).shape == (6, 3, 4, 4)
    assert visual_token_count(336, 336, 14) == 576
    assert visual_token_count(448, 448, 14, merge=2) == 256
    assert visual_token_count(448, 448, 14, n_queries=64) == 64


def _vlm(vocab=16, d_v=16, d_llm=24, image_token_id=15):
    vision = ToyVisionEncoder(image_size=8, patch=4, in_channels=1, d_v=d_v, depth=1, n_heads=2)
    lm = TinyCausalLM(vocab=vocab, max_len=32, d_llm=d_llm, depth=2, n_heads=4)
    return MiniVLM(vision, d_v, lm, d_llm, image_token_id)


def test_merge_and_forward_shapes():
    model = _vlm()
    imgs = torch.randn(2, 1, 8, 8)
    ids = torch.tensor([[1, 15, 3, 4, 5], [2, 15, 6, 7, 8]])  # (B, T=5), image token at col 1
    logits, is_visual = model(imgs, ids)
    assert logits.shape == (2, 5 - 1 + 4, 16)
    assert is_visual.sum(1).tolist() == [4, 4]
    assert bool(is_visual[0, 1]) and bool(is_visual[0, 4]) and not bool(is_visual[0, 5])
    text = model.lm.tok(ids)
    merged, pos, _ = merge_visual_tokens(text, ids, torch.zeros(2, 4, 24), 15)
    assert torch.equal(merged[:, 0], text[:, 0]) and torch.equal(merged[:, 5], text[:, 2])
    assert pos.shape == (2, 8) and pos[0, -1] == 7


def test_vlm_overfits_captions():
    torch.manual_seed(0)
    model = _vlm()
    n = 8
    imgs = torch.randn(n, 1, 8, 8) * 0.2
    ids = torch.zeros(n, 5, dtype=torch.long)
    for i in range(n):
        k = i % 4
        r, c = (k // 2) * 4, (k % 2) * 4
        imgs[i, 0, r : r + 4, c : c + 4] += 3.0
        ids[i] = torch.tensor([1, 15, 3, 4 + k, 4 + k])  # caption depends on the image
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad()
        logits, is_visual = model(imgs, ids)
        loss = vlm_lm_loss(logits, ids, is_visual, 15)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()
    logits, is_visual = model(imgs, ids)
    # position that predicts the class-dependent token: merged index of ids[:, 3] is 1 + 4 + 1 = 6 -> pred at 5
    pred = logits[:, 5].argmax(-1)
    assert torch.equal(pred, ids[:, 3])
