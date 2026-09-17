"""SARSA / Q-learning / Double Q / Expected SARSA recover VI's optimal policy; max bias demo."""
import numpy as np

from mlbook.rl.dynamic_programming import greedy_policy, one_hot_policy, policy_evaluation, value_iteration
from mlbook.rl.envs import GridWorld
from mlbook.rl.q_learning import (
    double_q_learning,
    epsilon_greedy_action,
    expected_sarsa,
    maximisation_bias_demo,
    q_learning,
    sarsa,
)

GAMMA = 0.95


def _check(algo, n_episodes=3000, tol=0.05, cells=((3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2))):
    env = GridWorld()
    V_star, pi_star, _ = value_iteration(env.P, env.R, GAMMA)
    Q = algo(env, GAMMA, n_episodes, alpha=0.1, eps=0.1, rng=np.random.default_rng(0))
    pi = greedy_policy(Q)
    V = policy_evaluation(env.P, env.R, one_hot_policy(pi, 4), GAMMA)
    start = env.to_index(env.start)
    assert V[start] > V_star[start] - tol, (V[start], V_star[start])
    # the greedy policy must match VI on these cells (the optimal path for off-policy methods)
    for cell in cells:
        assert pi[env.to_index(cell)] == pi_star[env.to_index(cell)], cell


# On-policy methods with eps = 0.1 converge to the best eps-SOFT policy: their greedy policy is
# good (within ~0.1 of V* at the start) but may take a route that keeps clear of the pit.
ON_POLICY_TOL, ON_POLICY_CELLS = 0.12, ((0, 0), (0, 1), (0, 2))


def test_epsilon_greedy_action_breaks_ties_randomly():
    rng = np.random.default_rng(0)
    Q = np.zeros((1, 4))
    picks = {epsilon_greedy_action(Q, 0, 0.0, rng) for _ in range(50)}
    assert picks == {0, 1, 2, 3}


def test_sarsa_finds_near_optimal_policy():
    _check(sarsa, tol=ON_POLICY_TOL, cells=ON_POLICY_CELLS)


def test_q_learning_finds_optimal_policy():
    _check(q_learning)


def test_expected_sarsa_finds_near_optimal_policy():
    _check(expected_sarsa, tol=ON_POLICY_TOL, cells=ON_POLICY_CELLS)


def test_double_q_learning_finds_optimal_policy():
    _check(double_q_learning, n_episodes=5000)


def test_maximisation_bias_is_positive_and_double_estimator_removes_it():
    single, double = maximisation_bias_demo(2000, 5, np.random.default_rng(0))
    assert single > 0.4 and abs(double) < 0.1
