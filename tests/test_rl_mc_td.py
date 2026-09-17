"""MC, TD(0), n-step and TD(lambda) prediction converge toward the DP value; MC control finds a good policy."""
import numpy as np

from mlbook.rl.dynamic_programming import greedy_policy, one_hot_policy, policy_evaluation, value_iteration
from mlbook.rl.envs import GridWorld
from mlbook.rl.mc_td import (
    discounted_returns,
    mc_control,
    mc_evaluation,
    n_step_td_evaluation,
    td0_evaluation,
    td_lambda_evaluation,
)

GAMMA = 0.95


def _setup():
    env = GridWorld()
    pi = np.full((16, 4), 0.25)  # uniform random policy visits every state
    V_true = policy_evaluation(env.P, env.R, pi, GAMMA)
    start = env.to_index(env.start)
    return env, pi, V_true, start


def test_discounted_returns():
    G = discounted_returns([1.0, 2.0, 3.0], 0.5)
    assert np.allclose(G, [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3.0])


def test_mc_evaluation_matches_dp_at_start():
    env, pi, V_true, start = _setup()
    V = mc_evaluation(env, pi, GAMMA, 1500, np.random.default_rng(0))
    assert abs(V[start] - V_true[start]) < 0.08


def test_mc_evaluation_with_constant_alpha_also_converges():
    """A constant step tracks instead of averaging, so it needs a looser tolerance."""
    env, pi, V_true, start = _setup()
    V = mc_evaluation(env, pi, GAMMA, 1500, np.random.default_rng(0), alpha=0.05)
    assert abs(V[start] - V_true[start]) < 0.12


def test_td0_evaluation_matches_dp_at_start():
    env, pi, V_true, start = _setup()
    V = td0_evaluation(env, pi, GAMMA, 1500, alpha=0.05, rng=np.random.default_rng(0))
    assert abs(V[start] - V_true[start]) < 0.08


def test_n_step_td_evaluation_matches_dp_at_start():
    env, pi, V_true, start = _setup()
    V = n_step_td_evaluation(env, pi, GAMMA, n=3, n_episodes=1500, alpha=0.05, rng=np.random.default_rng(0))
    assert abs(V[start] - V_true[start]) < 0.08


def test_td_lambda_evaluation_matches_dp_at_start():
    env, pi, V_true, start = _setup()
    V = td_lambda_evaluation(env, pi, GAMMA, lam=0.8, n_episodes=1500, alpha=0.05, rng=np.random.default_rng(0))
    assert abs(V[start] - V_true[start]) < 0.08


def test_mc_control_reaches_near_optimal_start_value():
    env = GridWorld()
    V_star, _, _ = value_iteration(env.P, env.R, GAMMA)
    Q, _ = mc_control(env, GAMMA, 3000, eps=0.1, rng=np.random.default_rng(0))
    V_greedy = policy_evaluation(env.P, env.R, one_hot_policy(greedy_policy(Q), 4), GAMMA)
    start = env.to_index(env.start)
    # on-policy control with eps = 0.1 converges to the best eps-soft policy, whose greedy version
    # is good but not necessarily optimal (it avoids cells next to the pit): allow a small gap
    assert V_greedy[start] > V_star[start] - 0.12
