import torch

from mlbook.interp import sparse_autoencoder as sae_mod


def test_sae_forward_shapes_and_unit_norm_decoder():
    torch.manual_seed(0)
    sae = sae_mod.SparseAutoencoder(d=8, n_features=32)
    x = torch.randn(5, 8)
    x_hat, f = sae(x)
    assert x_hat.shape == (5, 8) and f.shape == (5, 32) and (f >= 0).all()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(32), atol=1e-6)


def test_sae_loss_matches_manual():
    x, x_hat, f = torch.ones(2, 3), torch.zeros(2, 3), torch.full((2, 4), 0.5)
    assert torch.isclose(sae_mod.sae_loss(x, x_hat, f, 0.1), torch.tensor(3.0 + 0.1 * 2.0))


def test_train_sae_recovers_superposed_features():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    X, dirs = sae_mod.make_superposition_data(n=4096, d=16, n_true=32, p_active=0.05)
    sae = sae_mod.SparseAutoencoder(16, 64)
    before = sae_mod.feature_recovery(sae, dirs).mean()
    sae_mod.train_sae(sae, X, l1_coeff=0.2, steps=600)
    after = sae_mod.feature_recovery(sae, dirs)
    assert after.mean() > before and after.mean() > 0.85
