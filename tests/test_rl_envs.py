"""GridWorld tensors are a valid MDP; the control toys behave as documented."""
import numpy as np

from mlbook.rl.envs import CorridorEnv, GridWorld, PointMass1D, run_episode


def test_gridworld_P_is_stochastic_and_terminals_absorb():
    env = GridWorld()
    assert env.P.shape == (16, 4, 16) and env.R.shape == (16, 4)
    assert np.allclose(env.P.sum(axis=2), 1.0)
    for cell in (env.goal, env.pit):
        s = env.to_index(cell)
        assert np.allclose(env.P[s, :, s], 1.0) and np.allclose(env.R[s], 0.0)


def test_gridworld_slip_and_reward_values():
    env = GridWorld(slip=0.2)
    s = env.to_index((0, 2))  # just left of the goal
    right = 1
    assert np.isclose(env.P[s, right, env.to_index((0, 3))], 0.8)  # intended move
    assert np.isclose(env.P[s, right, env.to_index((1, 2))], 0.1)  # slip down
    assert np.isclose(env.P[s, right, s], 0.1)  # slip up hits the border -> stay
    assert np.isclose(env.R[s, right], 0.8 * 1.0 + 0.2 * env.step_cost)


def test_gridworld_step_samples_from_P():
    env, rng = GridWorld(slip=0.0), np.random.default_rng(0)
    env.reset(rng)
    obs, r, done = env.step(0, rng)  # up from (3,0) -> (2,0)
    assert obs.shape == (16,) and obs.argmax() == env.to_index((2, 0)) and r == env.step_cost and not done


def test_point_mass_random_policy_is_bad_and_expert_is_good():
    rng = np.random.default_rng(0)
    env = PointMass1D()
    random_ret = np.mean([run_episode(env, lambda o: int(rng.integers(3)), rng) for _ in range(30)])
    pd = lambda o: int(np.clip(np.round(1 - np.sign(4 * o[0] + 2 * o[1])), 0, 2))  # hand-tuned PD controller
    pd_ret = np.mean([run_episode(env, pd, rng) for _ in range(30)])
    assert random_ret < -8 and pd_ret > -4.5  # the PD reference averages about -3.2


def test_corridor_expert_keeps_lane_and_random_does_not():
    rng = np.random.default_rng(0)
    env = CorridorEnv()
    expert = np.mean([run_episode(env, env.expert_action, rng) for _ in range(50)])
    rand = np.mean([run_episode(env, lambda o: int(rng.integers(3)), rng) for _ in range(50)])
    assert expert > 34 and rand < 20
