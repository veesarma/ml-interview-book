"""PPO: clipped-loss gradient regions, value loss, rollout buffer, an update that moves the policy, and learning."""
import numpy as np
import torch

from mlbook.rl.actor_critic import Actor, Critic
from mlbook.rl.envs import GridWorld
from mlbook.rl.ppo import PPOConfig, RolloutBuffer, collect_rollout, ppo_clip_loss, ppo_update, ppo_value_loss, train_ppo

torch.set_num_threads(1)
torch.optim.Adam([torch.zeros(1, requires_grad=True)])


def _grad(logp_new_val, logp_old_val, adv_val, eps=0.2):
    logp_new = torch.tensor([logp_new_val], requires_grad=True)
    loss = ppo_clip_loss(logp_new, torch.tensor([logp_old_val]), torch.tensor([adv_val]), eps)
    loss.backward()
    return float(logp_new.grad)


def test_ppo_clip_loss_gradient_inside_and_outside_clip_region():
    # ratio = 1 (first epoch): gradient of -r*A wrt logp is -r*A = -A
    assert np.isclose(_grad(0.0, 0.0, 2.0), -2.0)
    # A > 0 and ratio = 1.5 > 1 + eps: clipped branch is active -> zero gradient
    assert _grad(np.log(1.5), 0.0, 2.0) == 0.0
    # A > 0 and ratio = 0.5 < 1 - eps: min picks the unclipped branch -> gradient -r*A = -1.0
    assert np.isclose(_grad(np.log(0.5), 0.0, 2.0), -0.5 * 2.0)
    # A < 0 and ratio = 0.5 < 1 - eps: clipped -> zero gradient
    assert _grad(np.log(0.5), 0.0, -2.0) == 0.0
    # A < 0 and ratio = 1.5: unclipped branch is the min -> gradient -r*A = +3.0
    assert np.isclose(_grad(np.log(1.5), 0.0, -2.0), 1.5 * 2.0)


def test_ppo_clip_loss_is_pessimistic_bound():
    torch.manual_seed(0)
    logp_new, logp_old, adv = torch.randn(100), torch.randn(100), torch.randn(100)
    surrogate = -(torch.exp(logp_new - logp_old) * adv).mean()
    assert ppo_clip_loss(logp_new, logp_old, adv, 0.2) >= surrogate - 1e-6


def test_ppo_value_loss():
    assert torch.isclose(ppo_value_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 4.0])), torch.tensor(2.5))


def test_rollout_buffer_and_collect_rollout_fill_all_slots():
    torch.manual_seed(0)
    env = GridWorld()
    actor, critic = Actor(env.obs_dim, env.n_actions, 8), Critic(env.obs_dim, 8)
    buf = RolloutBuffer(64, env.obs_dim)
    rng = np.random.default_rng(0)
    state = {"obs": env.reset(rng), "ep_return": 0.0, "returns": []}
    last_value = collect_rollout(env, actor, critic, buf, rng, state)
    assert buf.ptr == 64 and isinstance(last_value, float)
    assert np.all(buf.obs.sum(axis=1) == 1.0)  # one-hot observations
    assert np.all(buf.logp <= 0.0) and set(np.unique(buf.actions)) <= {0, 1, 2, 3}


def test_ppo_update_increases_probability_of_high_advantage_actions():
    torch.manual_seed(0)
    obs_dim, A, N = 4, 3, 128
    actor, critic = Actor(obs_dim, A, 16), Critic(obs_dim, 16)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-2)
    buf = RolloutBuffer(N, obs_dim)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, obs_dim)).astype(np.float32)
    actions = rng.integers(0, A, size=N)
    with torch.no_grad():
        logp = torch.log_softmax(actor(torch.from_numpy(obs)), dim=1).gather(1, torch.from_numpy(actions)[:, None]).squeeze(1)
    for i in range(N):  # reward +1 when action 2 was taken, else 0; every step is its own episode
        buf.add(obs[i], int(actions[i]), float(logp[i]), 1.0 if actions[i] == 2 else 0.0, 0.0, True)
    p_before = torch.softmax(actor(torch.from_numpy(obs)), dim=1)[:, 2].mean()
    stats = ppo_update(actor, critic, opt, buf, 0.0, PPOConfig(n_epochs=3, minibatch_size=32, target_kl=None))
    p_after = torch.softmax(actor(torch.from_numpy(obs)), dim=1)[:, 2].mean()
    assert p_after > p_before and stats["epochs_run"] == 3 and 0.0 <= stats["clip_frac"] <= 1.0


def test_train_ppo_learns_gridworld():
    env = GridWorld()
    _, _, returns = train_ppo(env, n_iterations=10, cfg=PPOConfig(gamma=0.95, rollout_steps=256), seed=0)
    assert np.mean(returns[-20:]) > 0.3 and np.mean(returns[-20:]) > np.mean(returns[:10]) + 0.5
