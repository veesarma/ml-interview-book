import torch

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers

from mlbook.multimodal.swin_window import (  # noqa: E402
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


def test_region_ids_marks_wrapping_cells():
    ids = region_ids(6, 6, 2)
    assert ids[0, 0] == 3 and ids[0, 5] == 2 and ids[5, 0] == 1 and ids[5, 5] == 0


def test_shifted_mask_matches_brute_force():
    h = w = 12
    M, s = 4, 2
    mask = shifted_window_mask(h, w, M, s)
    nW = (h // M) * (w // M)
    assert mask.shape == (nW, 16, 16)
    # Independent brute force: after the roll, rolled cell (r, c) came from original
    # ((r + s) % h, (c + s) % w). Two cells in a window may attend iff their ORIGINAL
    # coordinates are within M-1 of each other on both axes (true spatial neighbours).
    for wi in range(nW):
        r0, c0 = (wi // (w // M)) * M, (wi % (w // M)) * M
        cells = [(r0 + i, c0 + j) for i in range(M) for j in range(M)]
        for a, (ra, ca) in enumerate(cells):
            for b, (rb, cb) in enumerate(cells):
                oa = ((ra + s) % h, (ca + s) % w)
                ob = ((rb + s) % h, (cb + s) % w)
                neighbours = abs(oa[0] - ob[0]) < M and abs(oa[1] - ob[1]) < M
                assert (mask[wi, a, b] == 0) == neighbours, (wi, a, b)
    assert torch.all(mask[0] == 0)  # top-left window: no wrapped cells
    assert torch.any(mask[nW - 1] < 0)  # bottom-right window mixes four regions
    assert torch.all(shifted_window_mask(h, w, M, 0) == 0)


def test_relative_position_index_range_and_symmetry():
    idx = relative_position_index(3)
    assert idx.shape == (9, 9)
    assert idx.min() == 0 and idx.max() == 24
    # offset (0,0) for every diagonal entry -> same index
    assert torch.all(idx.diagonal() == idx[0, 0])
    # pair (q, k) and (k, q) have opposite offsets -> mirrored index
    assert idx[0, 8] + idx[8, 0] == 2 * idx[0, 0]


def test_window_attention_respects_mask():
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


def test_swin_block_shapes_and_shift_roundtrip():
    x = torch.randn(2, 8, 8, 16)
    y = SwinBlock(16, 4, M=4, shift=0)(x)
    z = SwinBlock(16, 4, M=4, shift=2)(y)
    assert y.shape == z.shape == (2, 8, 8, 16)


def test_patch_merging():
    x = torch.randn(2, 8, 8, 16)
    m = PatchMerging(16)(x)
    assert m.shape == (2, 4, 4, 32)
