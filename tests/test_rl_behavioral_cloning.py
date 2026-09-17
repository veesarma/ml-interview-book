"""BC: expert data collection, the tabular learner, and an MLP clone that recovers the VI-optimal gridworld policy."""
import numpy as np
import torch

from mlbook.rl.behavioral_cloning import MLPPolicy, TabularPolicy, collect_expert_data, fit_bc_mlp
from mlbook.rl.dynamic_programming import value_iteration
from mlbook.rl.envs import CorridorEnv, GridWorld

torch.set_num_threads(1)
torch.optim.Adam([torch.zeros(1, requires_grad=True)])


def test_collect_expert_data_shapes_and_labels():
    env, rng = CorridorEnv(), np.random.default_rng(0)
    obs, actions = collect_expert_data(env, env.expert_action, 5, rng)
    assert obs.ndim == 2 and obs.shape[1] == 2 and actions.shape == (obs.shape[0],)
    assert set(np.unique(actions)) <= {0, 1, 2}


def test_tabular_policy_learns_seen_bins_and_falls_back_to_prior():
    rng = np.random.default_rng(0)
    pol = TabularPolicy(3, 0.5, rng)
    obs = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.7, 0.0], [0.8, 0.0], [0.9, 0.0]])
    acts = np.array([1, 1, 1, 0, 0, 2])
    pol.fit(obs, acts)
    assert pol.act(np.array([0.4, 0.0])) == 1 and pol.act(np.array([0.6, 0.0])) == 0
    assert pol.act(np.array([5.0, 5.0])) == 1  # unseen bin -> global mode (action 1 is the most frequent label)
    assert pol.n_bins() == 2


def test_mlp_bc_clones_optimal_gridworld_policy():
    env = GridWorld()
    _, pi_star, _ = value_iteration(env.P, env.R, 0.95)
    live = [s for s in range(env.n_states) if not env.is_terminal(s) and env.to_cell(s) not in env.walls]
    obs = np.stack([env.observe(s) for s in live])  # (N, S) one-hot
    actions = pi_star[live]  # (N,)
    torch.manual_seed(0)
    policy = MLPPolicy(env.obs_dim, env.n_actions, hidden=32)
    loss = fit_bc_mlp(policy, obs, actions, n_epochs=300, lr=1e-2)
    assert loss < 0.1
    assert all(policy.act(env.observe(s)) == pi_star[s] for s in live)
