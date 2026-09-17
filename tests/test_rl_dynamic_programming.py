"""Value iteration and policy iteration agree with each other and with the closed form."""
import numpy as np

from mlbook.rl.dynamic_programming import (
    greedy_policy,
    one_hot_policy,
    policy_evaluation,
    policy_evaluation_exact,
    policy_iteration,
    q_from_v,
    value_iteration,
)
from mlbook.rl.envs import GridWorld

GAMMA = 0.95


def test_q_from_v_matches_einsum():
    env = GridWorld()
    V = np.random.default_rng(0).normal(size=env.n_states)
    Q = q_from_v(env.P, env.R, V, GAMMA)
    assert Q.shape == (16, 4)
    assert np.allclose(Q, env.R + GAMMA * np.einsum("sat,t->sa", env.P, V))


def test_policy_evaluation_iterative_equals_exact():
    env = GridWorld()
    pi = np.full((16, 4), 0.25)  # uniform random policy
    V_iter = policy_evaluation(env.P, env.R, pi, GAMMA, tol=1e-12)
    V_exact = policy_evaluation_exact(env.P, env.R, pi, GAMMA)
    assert np.allclose(V_iter, V_exact, atol=1e-8)


def test_value_iteration_is_a_fixed_point():
    env = GridWorld()
    V, pi, _ = value_iteration(env.P, env.R, GAMMA, tol=1e-12)
    assert np.allclose(V, np.max(q_from_v(env.P, env.R, V, GAMMA), axis=1), atol=1e-9)
    assert pi.shape == (16,) and pi[env.to_index((0, 2))] == 1  # step right into the goal


def test_policy_iteration_agrees_with_value_iteration():
    env = GridWorld()
    V_vi, pi_vi, _ = value_iteration(env.P, env.R, GAMMA, tol=1e-12)
    V_pi, pi_pi, n = policy_iteration(env.P, env.R, GAMMA)
    assert np.allclose(V_vi, V_pi, atol=1e-8)
    assert n < 10  # PI needs only a handful of improvement steps
    live = [s for s in range(16) if not env.is_terminal(s) and env.to_cell(s) not in env.walls]
    assert np.array_equal(pi_vi[live], pi_pi[live])


def test_greedy_policy_improves_value():
    env = GridWorld()
    pi0 = np.full((16, 4), 0.25)
    V0 = policy_evaluation(env.P, env.R, pi0, GAMMA)
    pi1 = one_hot_policy(greedy_policy(q_from_v(env.P, env.R, V0, GAMMA)), 4)
    V1 = policy_evaluation(env.P, env.R, pi1, GAMMA)
    assert np.all(V1 >= V0 - 1e-9) and V1[env.to_index(env.start)] > V0[env.to_index(env.start)]
