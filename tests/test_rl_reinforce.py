"""REINFORCE: reward-to-go, the baseline identity E[grad log pi * b] = 0, and learning on GridWorld."""
import numpy as np
import torch
from torch.distributions import Categorical

from mlbook.rl.envs import GridWorld
from mlbook.rl.reinforce import PolicyNetwork, reinforce_loss, rewards_to_go, sample_action, train_reinforce

torch.set_num_threads(1)
torch.optim.Adam([torch.zeros(1, requires_grad=True)])


def test_rewards_to_go():
    G = rewards_to_go(np.array([1.0, 0.0, 2.0], dtype=np.float32), 0.5)
    assert np.allclose(G, [1 + 0 + 0.25 * 2, 0 + 0.5 * 2, 2.0])


def test_sample_action_matches_softmax_frequencies():
    torch.manual_seed(0)
    policy = PolicyNetwork(3, 4, hidden=8)
    obs = np.ones(3, dtype=np.float32)
    rng = np.random.default_rng(0)
    counts = np.bincount([sample_action(policy, obs, rng)[0] for _ in range(4000)], minlength=4) / 4000
    probs = torch.softmax(policy(torch.from_numpy(obs)[None]), dim=1)[0].detach().numpy()
    assert np.allclose(counts, probs, atol=0.03)


def test_baseline_term_has_zero_expected_gradient():
    """sum_a pi(a|s) grad log pi(a|s) = grad sum_a pi = 0, so a constant baseline adds no bias."""
    torch.manual_seed(0)
    policy = PolicyNetwork(3, 4, hidden=8)
    obs = torch.randn(1, 3)
    logits = policy(obs)  # (1, A)
    probs = torch.softmax(logits, dim=1)[0].detach()  # (A,)
    total = [torch.zeros_like(p) for p in policy.parameters()]
    for a in range(4):  # exact expectation over actions
        logp = Categorical(logits=policy(obs)).log_prob(torch.tensor([a]))
        grads = torch.autograd.grad(logp.sum(), list(policy.parameters()))
        for acc, g in zip(total, grads):
            acc += probs[a] * g
    assert all(torch.allclose(t, torch.zeros_like(t), atol=1e-6) for t in total)


def test_reinforce_loss_value_and_sign():
    logp = torch.tensor([-1.0, -2.0], requires_grad=True)
    G = torch.tensor([3.0, 1.0])
    loss = reinforce_loss(logp, G, baseline=torch.tensor([1.0, 1.0]))
    assert torch.isclose(loss, torch.tensor(-((-1.0) * 2.0 + (-2.0) * 0.0) / 2))
    loss.backward()
    assert torch.allclose(logp.grad, torch.tensor([-1.0, 0.0]))  # d/dlogp of -(logp * w)/T


def test_train_reinforce_learns_gridworld():
    _, returns = train_reinforce(GridWorld(), n_episodes=250, gamma=0.95, lr=1e-2, seed=0)
    assert np.mean(returns[-30:]) > 0.5 and np.mean(returns[-30:]) > np.mean(returns[:30]) + 0.5
