import numpy as np

from mlbook.systems import speculative_decoding as sd


def test_acceptance_rate_is_one_minus_total_variation():
    p, q = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    assert np.isclose(sd.acceptance_rate(p, q), 1 - 0.5 * np.abs(p - q).sum())


def test_residual_distribution_is_positive_part_normalised():
    p, q = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    assert np.allclose(sd.residual_distribution(p, q), [1.0, 0.0, 0.0])


def test_speculative_step_first_token_marginal_matches_target():
    rng = np.random.default_rng(0)
    P, Q = sd.make_toy_models(V=6, temperature_draft=2.0, rng=rng)
    n = 20000
    counts = np.zeros(6)
    for _ in range(n):
        counts[sd.speculative_step(0, P, Q, k=3, rng=rng)[0]] += 1
    tv = 0.5 * np.abs(counts / n - P[0]).sum()
    assert tv < 0.02  # sampling noise at n = 20000 is ~ 0.005


def test_speculative_step_second_token_conditional_matches_target():
    rng = np.random.default_rng(1)
    P, Q = sd.make_toy_models(V=5, temperature_draft=3.0, rng=rng)
    n = 20000
    pair_counts = np.zeros((5, 5))
    for _ in range(n):
        out = sd.speculative_step(2, P, Q, k=2, rng=rng)
        if len(out) >= 2:
            pair_counts[out[0], out[1]] += 1
    cond = pair_counts / np.maximum(pair_counts.sum(axis=1, keepdims=True), 1)
    tv = 0.5 * np.abs(cond - P).sum(axis=1).max()
    assert tv < 0.05


def test_expected_tokens_per_call_and_generate():
    assert np.isclose(sd.expected_tokens_per_call(0.0, 4), 1.0)
    assert np.isclose(sd.expected_tokens_per_call(1.0, 4), 5.0)
    rng = np.random.default_rng(2)
    P, Q = sd.make_toy_models(V=8, temperature_draft=1.2, rng=rng)
    toks, calls = sd.generate(P, Q, 0, 500, k=4, rng=rng)
    assert len(toks) == 500 and 1.5 < len(toks) / calls <= 5.0
