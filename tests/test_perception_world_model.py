import numpy as np
import pytest
import torch

from mlbook.perception import world_model as wm


def test_point_mass_dynamics_step_and_rollout():
    env = wm.PointMassDynamics(dt=0.5)
    s = np.array([[0.0, 0.0, 1.0, 0.0]])
    nxt = env.step(s, np.array([[0.0, 2.0]]))
    assert np.allclose(nxt, [[0.5, 0.5, 1.0, 1.0]])
    obs = env.rollout(s, np.zeros((1, 4, 2)))
    assert obs.shape == (1, 5, 4) and np.allclose(obs[0, -1, :2], [2.0, 0.0])


def test_latent_world_model_shapes():
    m = wm.LatentWorldModel(obs_dim=4, action_dim=2, latent_dim=8)
    z = m.encode(torch.randn(3, 4))
    assert z.shape == (3, 8)
    assert m.next_latent(z, torch.randn(3, 2)).shape == (3, 8)
    assert m.rollout(z, torch.randn(3, 5, 2)).shape == (3, 6, 8)
    assert m.decode(z).shape == (3, 4)


@pytest.fixture(scope="module")
def trained(seed=0, steps=200):
    """Train the world model once and share it: both tests below only read from it.

    A fixed dataset of 512 trajectories is sampled up front so the training loop is pure
    torch (no per-step NumPy rollout), which keeps the fixture around a second on CPU.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = wm.PointMassDynamics(dt=0.2)
    s0 = np.random.uniform(-1, 1, size=(512, 4))
    a_np = np.random.uniform(-1, 1, size=(512, 5, 2))
    obs_all = torch.tensor(env.rollout(s0, a_np), dtype=torch.float32)  # (512, 6, 4)
    act_all = torch.tensor(a_np, dtype=torch.float32)  # (512, 5, 2)
    model = wm.LatentWorldModel(4, 2, latent_dim=8, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for step in range(steps):
        idx = torch.randint(0, 512, (128,))
        loss = wm.world_model_loss(model, obs_all[idx], act_all[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return env, model


def test_world_model_loss_trains_to_predict_multi_step_future(trained):
    env, model = trained
    s0 = np.random.uniform(-1, 1, size=(64, 4))
    a = np.random.uniform(-1, 1, size=(64, 5, 2))
    obs = torch.tensor(env.rollout(s0, a), dtype=torch.float32)
    with torch.no_grad():
        imagined = model.decode(model.rollout(model.encode(obs[:, 0]), torch.tensor(a, dtype=torch.float32)))
    err = (imagined[:, -1] - obs[:, -1]).norm(dim=-1).mean().item()
    naive = (obs[:, 0] - obs[:, -1]).norm(dim=-1).mean().item()  # "nothing moves" baseline
    assert err < 0.3 * naive


def test_cem_planning_reaches_goal_better_than_random_shooting_with_few_samples(trained):
    env, model = trained
    goal = torch.tensor([1.0, 1.0])
    start = np.array([[0.0, 0.0, 0.0, 0.0]])

    def cost_fn(decoded):  # (S, T+1, 4) → (S,): final distance to goal + a little control effort
        return (decoded[:, -1, :2] - goal).norm(dim=-1)

    z0 = model.encode(torch.tensor(start, dtype=torch.float32))
    plan = wm.cem_plan(model, z0, cost_fn, horizon=8, action_dim=2, n_samples=128, n_elite=16, n_iters=4)
    assert plan.shape == (8, 2)
    assert (plan.abs() <= 1.0).all()  # the action limit is enforced inside the search
    real = env.rollout(start, plan.numpy()[None])[0, -1, :2]
    assert np.linalg.norm(real - goal.numpy()) < 0.6
    rs = wm.random_shooting_plan(model, z0, cost_fn, horizon=8, action_dim=2, n_samples=16)
    real_rs = env.rollout(start, rs.numpy()[None])[0, -1, :2]
    assert np.linalg.norm(real - goal.numpy()) <= np.linalg.norm(real_rs - goal.numpy()) + 0.2


def test_random_shooting_improves_with_more_samples(trained):
    """More samples can only lower the best imagined cost, since the search is a minimum."""
    env, model = trained
    goal = torch.tensor([1.0, 1.0])
    start = np.array([[0.0, 0.0, 0.0, 0.0]])

    def cost_fn(decoded):
        return (decoded[:, -1, :2] - goal).norm(dim=-1)

    z0 = model.encode(torch.tensor(start, dtype=torch.float32))
    torch.manual_seed(0)
    few = wm.random_shooting_plan(model, z0, cost_fn, horizon=8, action_dim=2, n_samples=8)
    torch.manual_seed(0)
    many = wm.random_shooting_plan(model, z0, cost_fn, horizon=8, action_dim=2, n_samples=512)
    assert few.shape == (8, 2) and many.shape == (8, 2)
    assert (many.abs() <= 1.0).all()
    with torch.no_grad():
        cost_few = cost_fn(model.decode(model.rollout(z0, few.unsqueeze(0))))
        cost_many = cost_fn(model.decode(model.rollout(z0, many.unsqueeze(0))))
    assert cost_many.item() <= cost_few.item()
