import numpy as np
from scipy.special import comb

from mlbook.evaluation import llm_metrics as lm


def test_exact_match_normalisation():
    assert lm.exact_match("The Eiffel Tower!", "eiffel tower") == 1.0
    assert lm.exact_match("Paris", "London") == 0.0


def test_pass_at_k_matches_combinatorial_formula():
    for n, c, k in [(10, 3, 1), (10, 3, 5), (20, 0, 5), (5, 5, 2), (7, 2, 7)]:
        expected = 1 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0
        assert np.isclose(lm.pass_at_k(n, c, k), expected)
    assert np.isclose(lm.pass_at_k(10, 3, 1), 0.3)  # pass@1 = c/n


def test_bradley_terry_recovers_ordering():
    true_logit = np.array([1.0, 0.0, -1.0])
    rng = np.random.default_rng(0)
    wins = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                p = 1 / (1 + np.exp(true_logit[j] - true_logit[i]))
                wins[i, j] = rng.binomial(400, p)
    s = lm.bradley_terry(wins)
    assert s[0] > s[1] > s[2] and np.isclose(s.mean(), 0.0)
    assert np.allclose(s - s[1], true_logit, atol=0.25)


def test_elo_update_symmetric():
    ra, rb = lm.elo_update(1500, 1500, 1.0, k=32)
    assert np.isclose(ra, 1516) and np.isclose(rb, 1484)


def test_position_consistency_and_kappa():
    ab = np.array(["A", "B", "tie", "A"])
    ba = np.array(["B", "A", "tie", "A"])  # last one flips with position -> inconsistent
    assert np.isclose(lm.position_consistency(ab, ba), 0.75)
    a = np.array([1, 1, 0, 0]); b = np.array([1, 1, 0, 0])
    assert lm.cohens_kappa(a, b) == 1.0
    out = lm.judge_agreement(np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0]))
    assert np.isclose(out["agreement"], 0.75) and np.isclose(out["kappa"], 0.5)
