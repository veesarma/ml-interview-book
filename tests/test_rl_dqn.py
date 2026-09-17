"""DQN components (network, buffer, loss, schedule) and end-to-end learning on PointMass1D."""
import copy

import numpy as np
import torch

from mlbook.rl.dqn import (
    DQNConfig,
    DuelingQNetwork,
    QNetwork,
    ReplayBuffer,
    dqn_loss,
    epsilon_schedule,
    greedy_action,
    train_dqn,
)
from mlbook.rl.envs import PointMass1D, run_episode

torch.set_num_threads(1)  # tiny nets: multi-threading only adds contention
torch.optim.Adam([torch.zeros(1, requires_grad=True)])  # pay torch's one-off optimizer import here, not in a test


def test_qnetwork_shapes():
    net = QNetwork(2, 3, hidden=16)
    assert net(torch.zeros(5, 2)).shape == (5, 3)


def test_dueling_advantages_are_mean_centred():
    net = DuelingQNetwork(2, 3, hidden=16)
    obs = torch.randn(4, 2)
    q = net(obs)
    v = net.value(torch.relu(net.trunk(obs)))  # (4, 1)
    assert torch.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-6)


def test_replay_buffer_wraps_and_samples_valid_transitions():
    buf = ReplayBuffer(capacity=4, obs_dim=2)
    for i in range(6):
        buf.add(np.array([i, i], dtype=np.float32), i % 3, float(i), np.array([i + 1, i + 1], dtype=np.float32), i == 5)
    assert buf.size == 4 and buf.ptr == 2
    obs, a, r, nxt, d = buf.sample(8, np.random.default_rng(0))
    assert obs.shape == (8, 2) and a.shape == (8,) and r.shape == (8,) and nxt.shape == (8, 2) and d.shape == (8,)
    assert torch.all(r >= 2)  # entries 0 and 1 were overwritten
    assert torch.all(nxt[:, 0] == obs[:, 0] + 1)


def test_dqn_loss_matches_manual_target():
    torch.manual_seed(0)
    q, tgt = QNetwork(2, 3, 8), QNetwork(2, 3, 8)
    obs, nxt = torch.randn(6, 2), torch.randn(6, 2)
    a = torch.tensor([0, 1, 2, 0, 1, 2])
    r = torch.randn(6)
    d = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    gamma = 0.9
    y = r + gamma * (1 - d) * tgt(nxt).max(dim=1).values
    q_sa = q(obs).gather(1, a[:, None]).squeeze(1)
    expected = torch.nn.functional.smooth_l1_loss(q_sa, y.detach())
    assert torch.isclose(dqn_loss(q, tgt, (obs, a, r, nxt, d), gamma), expected)
    # Double DQN: evaluate the online argmax with the target net
    a_star = q(nxt).argmax(dim=1, keepdim=True)
    y2 = r + gamma * (1 - d) * tgt(nxt).gather(1, a_star).squeeze(1)
    expected2 = torch.nn.functional.smooth_l1_loss(q_sa, y2.detach())
    assert torch.isclose(dqn_loss(q, tgt, (obs, a, r, nxt, d), gamma, double=True), expected2)


def test_dqn_loss_has_no_gradient_through_target():
    torch.manual_seed(0)
    q, tgt = QNetwork(2, 3, 8), QNetwork(2, 3, 8)
    batch = (torch.randn(4, 2), torch.tensor([0, 1, 2, 1]), torch.randn(4), torch.randn(4, 2), torch.zeros(4))
    dqn_loss(q, tgt, batch, 0.9).backward()
    assert all(p.grad is None for p in tgt.parameters())
    assert all(p.grad is not None for p in q.parameters())


def test_epsilon_schedule_is_linear_then_flat():
    assert epsilon_schedule(0, 1.0, 0.1, 100) == 1.0
    assert np.isclose(epsilon_schedule(50, 1.0, 0.1, 100), 0.55)
    assert np.isclose(epsilon_schedule(500, 1.0, 0.1, 100), 0.1)


def test_train_dqn_reaches_return_threshold():
    env = PointMass1D()
    q_net, returns = train_dqn(env, n_episodes=80, cfg=DQNConfig(), seed=0)
    rng = np.random.default_rng(1)
    greedy = np.mean([run_episode(env, lambda o: greedy_action(q_net, o), rng) for _ in range(20)])
    assert np.mean(returns[:10]) < -10  # random-ish start
    assert greedy > -6, greedy  # a random policy scores about -17; a PD controller about -2
