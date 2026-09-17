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


def _bimodal_batch(n=32, horizon=6):
    """A perfectly ambiguous junction: identical history, half the futures turn left, half right."""
    hist = torch.zeros(n, 4, 2)
    hist[:, :, 0] = torch.linspace(-3, 0, 4)  # everyone approaches the same junction identically
    others = torch.zeros(n, 1, 4, 2)
    turn_left = torch.arange(n) % 2 == 0
    gt = torch.zeros(n, horizon, 2)
    gt[:, :, 0] = torch.linspace(1, 6, horizon)
    gt[turn_left, :, 1] = torch.linspace(0.5, 3, horizon)
    gt[~turn_left, :, 1] = -torch.linspace(0.5, 3, horizon)
    return hist, others, gt


def _train(model, hist, others, gt, loss_fn, steps=120, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred, logits = model(hist, others)
        loss, _ = loss_fn(pred, logits, gt)
        loss.backward()
        opt.step()
    return model(hist, others)


def test_kmeans_anchors_recover_the_two_manoeuvres():
    torch.manual_seed(0)
    _, _, gt = _bimodal_batch()
    anchors = tp.kmeans_trajectory_anchors(gt, num_anchors=2, n_iters=20)
    assert anchors.shape == (2, 6, 2)
    # One anchor turns left (final y > 0), the other right; each matches a real future exactly.
    finals = sorted(a[-1, 1].item() for a in anchors)
    assert finals[0] < -2.5 and finals[1] > 2.5
    assign = tp.anchor_assignment(gt, anchors)
    assert assign[::2].unique().numel() == 1 and assign[1::2].unique().numel() == 1
    assert assign[0] != assign[1]  # left-turners and right-turners land on different anchors


def test_plain_wta_collapses_to_the_mean_on_a_symmetric_input():
    """Documents the dead-mode failure that motivates anchors: the winner is chosen by the
    model's own output, so one mode wins both manoeuvres and is dragged to their average."""
    torch.manual_seed(0)
    hist, others, gt = _bimodal_batch()
    model = tp.TrajectoryPredictor(d_model=32, num_modes=4, horizon=6)
    pred, logits = _train(model, hist, others, gt, tp.winner_takes_all_loss)
    _, best = tp.winner_takes_all_loss(pred, logits, gt)
    assert best.unique().numel() == 1  # a single mode won every example; the rest are dead
    winner = pred[0, best[0]]  # (T_f, 2) the collapsed trajectory
    # The true futures end at y = +3 and y = -3. The winner ends strictly between them,
    # which is the averaging failure. (Sweeping seeds 0-4, this collapses on 4 of the 5.)
    assert abs(winner[-1, 1].item()) < 2.0
    assert tp.min_ade(pred, gt).item() > 0.8  # neither manoeuvre is covered


def test_anchored_wta_covers_both_futures():
    torch.manual_seed(0)
    hist, others, gt = _bimodal_batch()
    anchors = tp.kmeans_trajectory_anchors(gt, num_anchors=4, n_iters=20)
    model = tp.TrajectoryPredictor(d_model=32, num_modes=4, horizon=6, anchors=anchors)
    loss_fn = lambda pred, logits, g: tp.anchor_wta_loss(pred, logits, g, anchors)  # noqa: E731
    pred, logits = _train(model, hist, others, gt, loss_fn)
    assert tp.min_ade(pred, gt).item() < 0.2  # both futures are covered by some mode
    assert tp.miss_rate(pred, gt, threshold=1.0).item() == 0.0
    # The two manoeuvres are covered by *different* modes, not by one averaged mode.
    best_left = tp.ade_per_mode(pred[::2], gt[::2]).argmin(dim=1)
    best_right = tp.ade_per_mode(pred[1::2], gt[1::2]).argmin(dim=1)
    assert best_left.unique().numel() == 1 and best_right.unique().numel() == 1
    assert best_left[0] != best_right[0]
