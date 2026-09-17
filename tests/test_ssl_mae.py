"""Masked autoencoder: patchify round-trip, masking bookkeeping, asymmetric encoder, masked loss."""
import torch

torch.set_num_threads(1)

from mlbook.ssl import mae as M


def test_patchify_unpatchify_round_trip_and_layout():
    img = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    patches = M.patchify(img, patch=4)
    assert patches.shape == (2, 4, 16)                    # N = (8/4)² = 4, D = 4·4·1
    assert torch.allclose(M.unpatchify(patches, 4, 1, 8, 8), img)
    # first patch is the top-left 4×4 block
    assert torch.allclose(patches[0, 0].reshape(4, 4), img[0, 0, :4, :4])


def test_patchify_multichannel():
    img = torch.randn(3, 3, 12, 12)
    p = M.patchify(img, patch=6)
    assert p.shape == (3, 4, 108)                          # 6·6·3 = 108
    assert torch.allclose(M.unpatchify(p, 6, 3, 12, 12), img)


def test_random_masking_keeps_the_right_count_and_restores_order():
    x = torch.randn(4, 16, 8)
    x_vis, mask, ids_restore = M.random_masking(x, mask_ratio=0.75)
    assert x_vis.shape == (4, 4, 8)                        # 25% kept
    assert torch.allclose(mask.sum(dim=1), torch.full((4,), 12.0))
    # every visible token is one of the original tokens, and exactly the unmasked ones
    for b in range(4):
        kept_rows = x[b][mask[b] == 0]                     # (4, 8)
        for row in x_vis[b]:
            assert torch.any((kept_rows - row).abs().sum(dim=1) < 1e-6)
    # ids_restore is a permutation
    assert torch.allclose(ids_restore.sort(dim=1).values, torch.arange(16).repeat(4, 1))


def test_random_masking_is_random_per_example():
    x = torch.randn(2, 16, 4)
    _, m1, _ = M.random_masking(x, 0.5)
    _, m2, _ = M.random_masking(x, 0.5)
    assert not torch.allclose(m1, m2)
    assert not torch.allclose(m1[0], m1[1])                # independent masks across the batch


def test_masked_reconstruction_loss_ignores_visible_patches():
    pred, target = torch.zeros(1, 4, 3), torch.zeros(1, 4, 3)
    target[0, 0] = 10.0                                    # a huge error on patch 0
    target[0, 1] = 2.0                                     # a smaller one on patch 1
    mask = torch.tensor([[0.0, 1.0, 0.0, 0.0]])            # only patch 1 is masked
    assert torch.isclose(M.masked_reconstruction_loss(pred, target, mask), torch.tensor(4.0))
    mask_both = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert torch.isclose(M.masked_reconstruction_loss(pred, target, mask_both), torch.tensor(52.0))


def test_mae_encoder_only_sees_visible_tokens():
    model = M.MAE(img_size=16, patch=4, in_chans=1, mask_ratio=0.75)
    latent, mask, ids_restore = model.encode(torch.randn(2, 1, 16, 16))
    assert latent.shape == (2, 4, 64)                       # 16 patches, 25% kept → 4 tokens
    assert model.decode(latent, ids_restore).shape == (2, 16, 16)  # (B, N, P·P·C)


def test_mae_trains_and_loss_is_on_masked_patches():
    torch.manual_seed(0)
    # structured images (horizontal ramps) so that masked patches are predictable from context
    base = torch.linspace(0, 1, 16)[None, :].repeat(16, 1)          # (16, 16)
    imgs = torch.stack([base * s for s in torch.linspace(0.5, 1.5, 16)])[:, None]  # (16, 1, 16, 16)
    model = M.MAE(img_size=16, patch=4, in_chans=1, d_enc=48, d_dec=32, mask_ratio=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = model(imgs)[0].item()
    for _ in range(200):
        loss, pred, mask = model(imgs)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3 * first
