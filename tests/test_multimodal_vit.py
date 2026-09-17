import torch
from torch import nn

from mlbook.multimodal.patch_embed import (
    PatchEmbedConv,
    PatchEmbedLinear,
    interpolate_pos_embed,
    patchify,
    sinusoidal_2d_pos_embed,
    unpatchify,
)
from mlbook.multimodal.vit import DistillableViT, TinyViT

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers


def test_patchify_roundtrip_and_count():
    x = torch.randn(2, 3, 8, 8)  # (B, C, H, W)
    t = patchify(x, 4)  # (B, 4, 48)
    assert t.shape == (2, 4, 48)
    assert torch.equal(unpatchify(t, 4, 3, 8, 8), x)
    # patch 0 is the top-left 4x4 block, laid out as (C, P, P)
    assert torch.equal(t[0, 0], x[0, :, :4, :4].reshape(-1))


def test_linear_patch_embed_equals_conv():
    lin = PatchEmbedLinear(3, 4, 16)
    conv = PatchEmbedConv(3, 4, 16)
    conv.load_from_linear(lin)
    x = torch.randn(2, 3, 12, 8)
    assert torch.allclose(lin(x), conv(x), atol=1e-5)


def test_sinusoidal_2d_pos_embed():
    pos = sinusoidal_2d_pos_embed(4, 6, 32)
    assert pos.shape == (24, 32)
    # row half is identical along a row of the grid; column half identical down a column
    grid = pos.view(4, 6, 32)
    assert torch.allclose(grid[1, 0, :16], grid[1, 5, :16])
    assert torch.allclose(grid[0, 2, 16:], grid[3, 2, 16:])


def test_interpolate_pos_embed():
    learned = torch.randn(16, 8)  # 4x4 grid
    big = interpolate_pos_embed(learned, (4, 4), (6, 6))
    assert big.shape == (36, 8)
    same = interpolate_pos_embed(learned, (4, 4), (4, 4))
    assert torch.allclose(same, learned, atol=1e-5)


def _synthetic(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 8, 8, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    # class k lights up patch k (top row) so a ViT must read positions
    for i in range(n):
        x[i, 0, 0:4, 4 * int(y[i]) % 8 : 4 * int(y[i]) % 8 + 4] += 3.0 * (1 + int(y[i]))
    return x, y


def test_vit_overfits_small_set():
    x, y = _synthetic(32)
    model = TinyViT(image_size=8, patch=4, in_channels=1, d=32, depth=2, n_heads=4, n_classes=3, n_registers=1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(100):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    acc = (model(x).argmax(-1) == y).float().mean().item()
    assert acc > 0.95, acc


def test_gap_and_distillation_heads():
    x, _ = _synthetic(4)
    gap = TinyViT(8, 4, 1, 16, 1, 2, 3, pool="gap")
    assert gap(x).shape == (4, 3)
    deit = DistillableViT(8, 4, 1, 16, 1, 2, 3)
    a, b = deit(x)
    assert a.shape == (4, 3) and b.shape == (4, 3)
