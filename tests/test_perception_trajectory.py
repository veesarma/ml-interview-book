import torch

from mlbook.perception import trajectory_prediction as tp


def test_to_agent_frame_puts_target_at_origin_heading_x():
    hist = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]])  # driving +y in the world
    others = torch.tensor([[[[1.0, 2.0], [1.0, 3.0]]]])  # one agent to the right/ahead
    h, o, R = tp.to_agent_frame(hist, others)
    assert torch.allclose(h[0, -1], torch.zeros(2), atol=1e-6)
    assert torch.allclose(h[0, 0], torch.tensor([-2.0, 0.0]), atol=1e-6)  # history lies along −x
    assert torch.allclose(o[0, 0, 0], torch.tensor([0.0, -1.0]), atol=1e-6)  # world +x is local −y
    assert torch.allclose(o[0, 0, 0] @ R[0].T + hist[0, -1], others[0, 0, 0], atol=1e-6)  # back to world


def test_polyline_encoder_is_permutation_invariant_over_time_steps():
    torch.manual_seed(0)
    enc = tp.PolylineEncoder(2, 8)
    x = torch.randn(2, 3, 5, 2)
    assert torch.allclose(enc(x), enc(x[:, :, [4, 0, 2, 1, 3]]))
    assert enc(x).shape == (2, 3, 8)


def test_social_attention_ignores_invalid_agents():
    torch.manual_seed(0)
    att = tp.SocialAttention(8)
    target, others = torch.randn(1, 8), torch.randn(1, 3, 8)
    valid = torch.tensor([[True, True, False]])
    out = att(target, others, valid)
    others2 = others.clone()
    others2[0, 2] = 100.0  # perturbing a masked agent changes nothing
    assert torch.allclose(out, att(target, others2, valid))
    assert torch.allclose(att(target, others, torch.zeros(1, 3, dtype=torch.bool)), target)


def test_winner_takes_all_picks_closest_mode_and_only_trains_it():
    torch.manual_seed(0)
    gt = torch.zeros(1, 4, 2)
    gt[0, :, 0] = torch.arange(1, 5).float()  # straight ahead
    pred = torch.zeros(1, 3, 4, 2, requires_grad=True)
    with torch.no_grad():
        pred[0, 0, :, 1] = 5.0  # far left
        pred[0, 1, :, 0] = torch.arange(1, 5).float() + 0.5  # close
        pred[0, 2, :, 1] = -5.0  # far right
    logits = torch.zeros(1, 3, requires_grad=True)
    loss, best = tp.winner_takes_all_loss(pred, logits, gt)
    assert best.tolist() == [1]
    loss.backward()
    assert pred.grad[0, 1].abs().sum() > 0 and pred.grad[0, 0].abs().sum() == 0 and pred.grad[0, 2].abs().sum() == 0
    assert logits.grad[0, 1] < 0 and logits.grad[0, 0] > 0  # CE pushes the winner's logit up


def test_metrics_minade_minfde_miss_rate():
    gt = torch.zeros(2, 3, 2)
    pred = torch.zeros(2, 2, 3, 2)
    pred[:, 0] = 10.0  # mode 0 is far; mode 1 exact
    assert tp.min_ade(pred, gt) == 0.0 and tp.min_fde(pred, gt) == 0.0
    assert tp.miss_rate(pred, gt, threshold=2.0) == 0.0
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    assert torch.isclose(tp.min_ade(pred, gt, k=1, logits=logits), torch.tensor(10.0 * 2 ** 0.5))
    pred[:, 1] = 3.0
    assert tp.miss_rate(pred, gt, threshold=2.0) == 1.0


def test_trajectory_predictor_learns_a_bimodal_future():
    torch.manual_seed(0)
    model = tp.TrajectoryPredictor(d_model=32, num_modes=4, horizon=6)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    hist = torch.zeros(64, 4, 2)
    hist[:, :, 0] = torch.linspace(-3, 0, 4)  # everyone approaches the same junction
    others = torch.zeros(64, 1, 4, 2)
    turn_left = torch.arange(64) % 2 == 0
    gt = torch.zeros(64, 6, 2)
    gt[:, :, 0] = torch.linspace(1, 6, 6)
    gt[turn_left, :, 1] = torch.linspace(0.5, 3, 6)
    gt[~turn_left, :, 1] = -torch.linspace(0.5, 3, 6)
    for _ in range(150):
        opt.zero_grad()
        pred, logits = model(hist, others)
        loss, _ = tp.winner_takes_all_loss(pred, logits, gt)
        loss.backward()
        opt.step()
    pred, logits = model(hist, others)
    assert tp.min_ade(pred, gt).item() < 0.3  # both futures are covered by some mode
    assert tp.miss_rate(pred, gt, threshold=1.0).item() == 0.0
