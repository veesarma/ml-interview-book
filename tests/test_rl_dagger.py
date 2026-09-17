"""On the corridor toy, behavioural cloning degrades badly and DAgger recovers most of the expert's return."""
import numpy as np

from mlbook.rl.behavioral_cloning import TabularPolicy, collect_expert_data
from mlbook.rl.dagger import dagger, make_tabular_learner, rollout_and_relabel
from mlbook.rl.envs import CorridorEnv, run_episode


def _mean_return(env, act, rng, n=100):
    return float(np.mean([run_episode(env, act, rng) for _ in range(n)]))


def test_rollout_and_relabel_uses_expert_labels():
    env, rng = CorridorEnv(), np.random.default_rng(0)
    always_left = type("L", (), {"act": staticmethod(lambda obs: 0)})()
    obs, labels, ret = rollout_and_relabel(env, always_left, env.expert_action, 3, rng, beta=0.0)
    assert obs.shape[0] == labels.shape[0] and ret < 15  # steering hard left crashes
    assert (labels == 2).mean() > 0.5  # the expert says "steer right" in most of those states


def test_bc_fails_and_dagger_recovers():
    rng = np.random.default_rng(0)
    env = CorridorEnv()
    expert_ret = _mean_return(env, env.expert_action, rng)
    obs, acts = collect_expert_data(env, env.expert_action, 20, rng)
    bc = TabularPolicy(3, 0.05, rng)
    bc.fit(obs, acts)
    bc_ret = _mean_return(env, bc.act, rng)
    learner, history = dagger(env, env.expert_action, make_tabular_learner(3, 0.05, rng), 8, 20, rng)
    dagger_ret = _mean_return(env, learner.act, rng)
    # measured: expert 40.0, BC 28.5, DAgger 38.1 on this seed sequence
    assert expert_ret > 38
    assert bc_ret < expert_ret - 8, (bc_ret, expert_ret)
    assert dagger_ret > bc_ret + 6 and dagger_ret > expert_ret - 4, (bc_ret, dagger_ret, expert_ret)
    assert len(history) == 8
