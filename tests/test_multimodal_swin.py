import torch

from mlbook.multimodal.swin_window import (
    PatchMerging,
    SwinBlock,
    WindowAttention,
    region_ids,
    relative_position_index,
    shifted_window_mask,
    window_partition,
    window_reverse,
)


def test_window_partition_roundtrip():
    x = torch.randn(2, 8, 8, 5)
    win = window_partition(x, 4)  # (2*4, 16, 5)
    assert win.shape == (8, 16, 5)
    assert torch.equal(window_reverse(win, 4, 8, 8), x)
    # window 0 of example 0 is the top-left 4x4 block in row-major order
    assert torch.equal(win[0], x[0, :4, :4].reshape(16, 5))


def test_shifted_mask_matches_brute_force():
    h = w = 8
    M, s = 4, 2
    mask = shifted_window_mask(h, w, M, s)
    assert mask.shape == (4, 16, 16)
    # brute force: two tokens may attend iff their un-shifted coordinates lie in the same region
    ids = region_ids(h, w, M, s)
    rolled = torch.roll(ids, (-s, -s), (0, 1))
    for wi in range(4):
        r0, c0 = (wi // 2) * M, (wi % 2) * M
        cells = [(r0 + i, c0 + j) for i in range(M) for j in range(M)]
        for a, (ra, ca) in enumerate(cells):
            for b, (rb, cb) in enumerate(cells):
                same = rolled[ra, ca] == rolled[rb, cb]
                assert (mask[wi, a, b] == 0) == bool(same)
    # the top-left window (all from one region after the roll) is fully unmasked
    assert torch.all(mask[0] == 0)
    # the bottom-right window mixes 4 regions -> has masked entries
    assert torch.any(mask[3] < 0)
    assert torch.all(shifted_window_mask(h, w, M, 0) == 0)


def test_relative_position_index_range_and_symmetry():
    idx = relative_position_index(3)
    assert idx.shape == (9, 9)
    assert idx.min() == 0 and idx.max() == 24
    # offset (0,0) for every diagonal entry -> same index
    assert torch.all(idx.diagonal() == idx[0, 0])


def test_window_attention_respects_mask():
    torch.manual_seed(0)
    attn = WindowAttention(d=8, n_heads=2, M=2)
    x = torch.randn(1 * 1, 4, 8)
    mask = torch.zeros(1, 4, 4)
    mask[0, 0, 1:] = -1e9  # token 0 may only see itself
    out = attn(x, mask)
    # changing tokens 1..3 must not change output of token 0
    x2 = x.clone()
    x2[:, 1:] += torch.randn(1, 3, 8)
    out2 = attn(x2, mask)
    assert torch.allclose(out[:, 0], out2[:, 0], atol=1e-5)
    assert not torch.allclose(out[:, 1], out2[:, 1])


def test_swin_block_and_merging_shapes():
    x = torch.randn(2, 8, 8, 16)
    y = SwinBlock(16, 4, M=4, shift=0)(x)
    z = SwinBlock(16, 4, M=4, shift=2)(y)
    assert z.shape == (2, 8, 8, 16)
    m = PatchMerging(16)(z)
    assert m.shape == (2, 4, 4, 32)
