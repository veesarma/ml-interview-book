"""Bandit solvers: UCB and Thompson sampling have sublinear regret; LinUCB learns a linear reward."""
import numpy as np

from mlbook.rl.bandits import (
    BernoulliBandit,
    EpsilonGreedy,
    LinUCB,
    ThompsonBeta,
    UCB1,
    run_bandit,
    run_linear_contextual,
)

PROBS = np.array([0.1, 0.5, 0.6, 0.9])


def _regret(solver_factory, horizon=4000, seed=0):
    rng = np.random.default_rng(seed)
    bandit = BernoulliBandit(PROBS, rng)
    return run_bandit(bandit, solver_factory(rng), horizon)


def test_epsilon_greedy_regret_grows_linearly():
    r = _regret(lambda rng: EpsilonGreedy(4, 0.1, rng))
    # with eps = 0.1 a quarter of the exploratory pulls hit each arm: slope >= 0.1 * mean gap
    expected_slope = 0.1 * np.mean(PROBS.max() - PROBS)
    assert r[-1] / 4000 > 0.7 * expected_slope


def test_ucb1_regret_is_sublinear():
    r = _regret(lambda rng: UCB1(4))
    assert r[-1] < r[2000] * 1.6  # second half adds far less than the first


def test_ucb1_pulls_every_arm_once_first():
    s = UCB1(4)
    pulled = []
    for _ in range(4):
        k = s.select()
        s.update(k, 0.0)
        pulled.append(k)
    assert sorted(pulled) == [0, 1, 2, 3]
    assert s.select() in range(4) and s.t == 5


def test_thompson_beta_posterior_update_and_low_regret():
    rng = np.random.default_rng(0)
    s = ThompsonBeta(2, rng)
    s.update(0, 1.0)
    s.update(0, 0.0)
    assert s.alpha[0] == 2 and s.beta[0] == 2 and s.alpha[1] == 1
    r = _regret(lambda rng: ThompsonBeta(4, rng))
    assert r[-1] < 60


def test_linucb_beats_random_on_linear_rewards():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(5, 4))  # (K, d)
    regret = run_linear_contextual(theta, LinUCB(5, 4, alpha=0.5), 1500, rng)
    assert regret[-1] < 0.15 * 1500  # random would give ~ 1 unit of regret per step
    assert regret[-1] - regret[1000] < regret[500]  # decelerating
