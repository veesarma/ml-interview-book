"""Tests for the toy MoE layer (Part VI, chapter 3)."""

import torch

from mlbook.llm.moe import Expert, MoELayer, TopKRouter, expert_usage_fraction, load_balancing_loss


def test_expert_shapes():
    e = Expert(8, 16)
    assert e(torch.randn(5, 8)).shape == (5, 8)


def test_router_topk_weights_sum_to_one():
    router = TopKRouter(d_model=8, n_experts=4, k=2)
    w, idx, p = router(torch.randn(10, 8))
    assert w.shape == (10, 2) and idx.shape == (10, 2) and p.shape == (10, 4)
    assert torch.allclose(w.sum(-1), torch.ones(10))
    assert torch.allclose(p.sum(-1), torch.ones(10))
    assert (idx[:, 0] != idx[:, 1]).all()


def test_load_balancing_loss_is_one_when_balanced_and_larger_when_not():
    E, k = 4, 1
    probs = torch.full((8, E), 1.0 / E)
    idx = torch.arange(8).remainder(E)[:, None]  # round-robin => f_i = 1/E
    assert torch.isclose(load_balancing_loss(probs, idx), torch.tensor(1.0))
    skewed_probs = torch.tensor([[0.97, 0.01, 0.01, 0.01]] * 8)
    all_to_zero = torch.zeros(8, 1, dtype=torch.long)
    assert load_balancing_loss(skewed_probs, all_to_zero) > 3.0  # ≈ E * 0.97


def test_moe_forward_shapes_and_capacity_drop():
    moe = MoELayer(d_model=8, d_ff=16, n_experts=4, k=2)
    y, aux = moe(torch.randn(2, 5, 8))
    assert y.shape == (2, 5, 8) and aux.ndim == 0
    assert moe.last_usage.sum() == 2 * 5 * 2  # every token reached both experts
    capped = MoELayer(d_model=8, d_ff=16, n_experts=4, k=2, capacity_factor=0.5)
    capped(torch.randn(2, 5, 8))
    assert capped.last_usage.max() <= 3  # ceil(0.5 * 10 * 2 / 4) = 3


def test_moe_with_k_equals_E_is_dense_weighted_sum():
    moe = MoELayer(d_model=8, d_ff=16, n_experts=3, k=3)
    x = torch.randn(1, 4, 8)
    y, _ = moe(x)
    w, idx, _ = moe.router(x.reshape(4, 8))
    ref = torch.zeros(4, 8)
    for e in range(3):
        we = w[torch.arange(4), (idx == e).float().argmax(-1)]  # weight assigned to expert e
        ref += we[:, None] * moe.experts[e](x.reshape(4, 8))
    assert torch.allclose(y.reshape(4, 8), ref, atol=1e-5)


def test_shared_expert_is_added_to_every_token():
    moe = MoELayer(d_model=8, d_ff=16, n_experts=2, k=1, n_shared=1)
    x = torch.randn(1, 3, 8)
    y_with, _ = moe(x)
    shared_out = moe.shared[0](x.reshape(3, 8))
    moe.shared = torch.nn.ModuleList([])
    y_without, _ = moe(x)
    assert torch.allclose(y_with.reshape(3, 8) - y_without.reshape(3, 8), shared_out, atol=1e-6)


def test_balance_loss_makes_routing_balanced():
    torch.manual_seed(1)
    E = 8
    router = TopKRouter(d_model=16, n_experts=E, k=2)
    with torch.no_grad():  # start badly imbalanced: one expert dominates
        router.gate.weight[0] += 3.0
    x = torch.randn(512, 16)
    _, idx0, _ = router(x)
    frac0 = expert_usage_fraction(idx0, E)
    opt = torch.optim.SGD(router.parameters(), lr=1.0)
    for _ in range(300):
        _, idx, probs = router(x)
        loss = load_balancing_loss(probs, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, idx1, probs1 = router(x)
    frac1 = expert_usage_fraction(idx1, E)
    assert frac1.max() < frac0.max()
    assert frac1.max() < 2.0 / E  # no expert takes more than 2x its fair share
    assert load_balancing_loss(probs1, idx1) < 1.15
