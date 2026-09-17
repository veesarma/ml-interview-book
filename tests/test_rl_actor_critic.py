"""Actor-critic losses reduce to the TD error, and the agent learns GridWorld."""
import numpy as np
import torch

from mlbook.rl.actor_critic import Actor, Critic, actor_critic_losses, train_actor_critic
from mlbook.rl.envs import GridWorld

torch.set_num_threads(1)
torch.optim.Adam([torch.zeros(1, requires_grad=True)])


def test_actor_critic_losses_match_manual_td_error():
    torch.manual_seed(0)
    actor, critic = Actor(2, 3, 8), Critic(2, 8)
    obs, nxt = torch.randn(5, 2), torch.randn(5, 2)
    a = torch.tensor([0, 2, 1, 1, 0])
    r = torch.randn(5)
    d = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    la, lc = actor_critic_losses(actor, critic, obs, a, r, nxt, d, 0.9)
    with torch.no_grad():
        delta = r + 0.9 * (1 - d) * critic(nxt) - critic(obs)
        logp = torch.log_softmax(actor(obs), dim=1).gather(1, a[:, None]).squeeze(1)
    assert torch.isclose(lc, (delta**2).mean())
    assert torch.isclose(la, -(logp * delta).mean())


def test_critic_loss_gradient_does_not_flow_into_target():
    torch.manual_seed(0)
    actor, critic = Actor(2, 3, 8), Critic(2, 8)
    obs = torch.randn(3, 2)
    la, lc = actor_critic_losses(actor, critic, obs, torch.tensor([0, 1, 2]), torch.ones(3), obs, torch.zeros(3), 0.9)
    lc.backward()
    # semi-gradient: d/dV of (r + gamma V(s') - V(s))^2 with s' = s must equal -2 delta (not 2(gamma-1) delta)
    with torch.no_grad():
        v = critic(obs)
        delta = 1.0 + 0.9 * v - v
    assert torch.allclose(critic.out.bias.grad, (-2 * delta).mean().reshape(1), atol=1e-5)


def test_train_actor_critic_learns_gridworld():
    _, _, returns = train_actor_critic(GridWorld(), n_episodes=250, gamma=0.95, seed=0)
    assert np.mean(returns[-30:]) > 0.5 and np.mean(returns[-30:]) > np.mean(returns[:30]) + 0.5
