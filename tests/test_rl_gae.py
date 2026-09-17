"""GAE equals TD(0) at lambda=0, MC-minus-V at lambda=1, and the lambda-weighted n-step mixture in between."""
import numpy as np

from mlbook.rl.gae import compute_gae, n_step_advantage

rng = np.random.default_rng(0)
T, GAMMA = 6, 0.9
rewards = rng.normal(size=T)
values = rng.normal(size=T)
dones = np.zeros(T)
dones[-1] = 1.0  # one complete episode
last_value = 123.0  # must be ignored because the episode terminated


def test_gae_lambda_zero_is_one_step_td_advantage():
    adv, ret = compute_gae(rewards, values, dones, last_value, GAMMA, lam=0.0)
    assert np.allclose(adv, n_step_advantage(rewards, values, dones, last_value, GAMMA, n=1))
    assert np.allclose(ret, adv + values)


def test_gae_lambda_one_is_monte_carlo_advantage():
    adv, _ = compute_gae(rewards, values, dones, last_value, GAMMA, lam=1.0)
    G = np.array([sum(GAMMA ** (k - t) * rewards[k] for k in range(t, T)) for t in range(T)])
    assert np.allclose(adv, G - values)


def test_gae_is_exponentially_weighted_n_step_mixture():
    lam = 0.7
    adv, _ = compute_gae(rewards, values, dones, last_value, GAMMA, lam)
    mixture = np.zeros(T)
    for n in range(1, T + 1):
        w = (1 - lam) * lam ** (n - 1) if n < T else lam ** (T - 1)  # tail mass goes to the full return
        mixture += w * n_step_advantage(rewards, values, dones, last_value, GAMMA, n)
    # the mixture with per-t truncation: for t, the max useful n is T - t; weights beyond collapse to the MC term
    exact = np.zeros(T)
    for t in range(T):
        acc = 0.0
        for n in range(1, T - t + 1):
            w = (1 - lam) * lam ** (n - 1) if n < T - t else lam ** (T - t - 1)
            acc += w * n_step_advantage(rewards, values, dones, last_value, GAMMA, n)[t]
        exact[t] = acc
    assert np.allclose(adv, exact)


def test_gae_bootstraps_truncated_segment_and_resets_at_done():
    r = np.array([1.0, 1.0, 1.0, 1.0])
    v = np.array([0.5, 0.5, 0.5, 0.5])
    d = np.array([0.0, 1.0, 0.0, 0.0])  # episode ends after step 1; segment truncated after step 3
    adv, _ = compute_gae(r, v, d, last_value=2.0, gamma=0.5, lam=0.5)
    delta = np.array([1 + 0.5 * 0.5 - 0.5, 1 + 0 - 0.5, 1 + 0.5 * 0.5 - 0.5, 1 + 0.5 * 2.0 - 0.5])
    expected = np.array([delta[0] + 0.25 * delta[1], delta[1], delta[2] + 0.25 * delta[3], delta[3]])
    assert np.allclose(adv, expected)
