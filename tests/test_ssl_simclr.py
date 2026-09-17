"""SimCLR / MoCo contrastive losses."""
import math

import torch

torch.set_num_threads(1)

import torch.nn.functional as F

from mlbook.ssl import simclr as S


def test_projection_head_shapes():
    head = S.ProjectionHead(d_h=16, d_z=8)
    assert head(torch.randn(5, 16)).shape == (5, 8)


def test_nt_xent_matches_the_loop_written_from_the_formula():
    z1, z2 = torch.randn(6, 4), torch.randn(6, 4)
    assert torch.isclose(S.nt_xent_loss(z1, z2, 0.2), S.nt_xent_loss_reference(z1, z2, 0.2), atol=1e-5)


def test_nt_xent_is_minimised_by_perfectly_aligned_views():
    z = F.normalize(torch.randn(8, 4), dim=1)
    aligned = S.nt_xent_loss(z, z.clone(), temperature=0.1)     # positives have cos = 1
    scrambled = S.nt_xent_loss(z, z.roll(1, dims=0), 0.1)       # positives are other images
    assert aligned < scrambled


def test_nt_xent_random_chance_value():
    """With orthogonal-ish embeddings every one of the 2B−1 candidates looks the same, so the
    loss approaches log(2B − 1)."""
    B, d = 16, 512
    z = F.normalize(torch.randn(2 * B, d), dim=1)               # near-orthogonal in high dim
    loss = S.nt_xent_loss(z[:B], z[B:], temperature=1.0)
    assert abs(loss.item() - math.log(2 * B - 1)) < 0.05


def test_nt_xent_temperature_sharpens():
    z1, z2 = torch.randn(8, 4), torch.randn(8, 4)
    # a colder temperature magnifies the (random) similarity gaps, so the loss moves
    assert S.nt_xent_loss(z1, z2, 1.0) != S.nt_xent_loss(z1, z2, 0.05)


def test_nt_xent_excludes_the_anchor_itself():
    """If z_i were left in its own denominator the loss could never fall below log 2."""
    z = F.normalize(torch.randn(4, 8), dim=1)
    assert S.nt_xent_loss(z, z.clone(), temperature=0.01).item() < 0.05


def test_info_nce_with_queue():
    q = F.normalize(torch.randn(4, 8), dim=1)
    loss_easy = S.info_nce_loss(q, q.clone(), torch.randn(64, 8), 0.07)   # positive is identical
    loss_hard = S.info_nce_loss(q, torch.randn(4, 8), torch.randn(64, 8), 0.07)
    assert loss_easy < loss_hard
    # at temperature 1 with near-orthogonal high-dimensional keys, all 1 + Q logits look alike
    # and the loss sits at the chance level log(1 + Q)
    q_hi = F.normalize(torch.randn(16, 512), dim=1)
    chance = S.info_nce_loss(q_hi, torch.randn(16, 512), torch.randn(256, 512), temperature=1.0)
    assert abs(chance.item() - math.log(1 + 256)) < 0.15


def test_cosine_similarity_matrix_diagonal_is_positives():
    z = torch.randn(5, 3)
    sim = S.cosine_similarity_matrix(z, z)
    assert sim.shape == (5, 5) and torch.allclose(sim.diag(), torch.ones(5), atol=1e-5)


def test_nt_xent_trains_two_views_to_agree():
    torch.manual_seed(0)
    x = torch.randn(32, 6)
    encoder = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
    head = S.ProjectionHead(16, 8)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=3e-3)
    def views():
        return x + 0.1 * torch.randn_like(x), x + 0.1 * torch.randn_like(x)
    v1, v2 = views()
    first = S.nt_xent_loss(head(encoder(v1)), head(encoder(v2)), 0.2).item()
    for _ in range(300):
        v1, v2 = views()
        loss = S.nt_xent_loss(head(encoder(v1)), head(encoder(v2)), 0.2)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first
    # positives end up more similar than negatives
    sim = S.cosine_similarity_matrix(head(encoder(v1)), head(encoder(v2)))
    assert sim.diag().mean() > sim.mean() + 0.2
