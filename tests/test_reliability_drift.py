import numpy as np
from scipy import stats

from mlbook.reliability import drift


def test_ks_statistic_and_pvalue_match_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=300), rng.normal(0.5, 1.2, size=250)
    ref = stats.ks_2samp(a, b, method="asymp")
    assert np.isclose(drift.ks_statistic(a, b), ref.statistic)
    assert np.isclose(drift.ks_pvalue(drift.ks_statistic(a, b), 300, 250), ref.pvalue, atol=1e-3)
    same = rng.normal(size=300)
    assert drift.ks_pvalue(drift.ks_statistic(a, same), 300, 300) > 0.05


def test_population_stability_index():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert drift.population_stability_index(ref, rng.normal(size=5000)) < 0.02
    assert drift.population_stability_index(ref, rng.normal(1.0, 1.0, size=5000)) > 0.25


def test_classifier_two_sample_test():
    rng = np.random.default_rng(0)
    X_ref = rng.normal(size=(400, 3))
    X_same = rng.normal(size=(400, 3))
    X_shift = rng.normal(size=(400, 3)) + np.array([1.5, 0, 0])
    assert drift.classifier_two_sample_test(X_ref, X_shift)["p_value"] < 0.01
    assert drift.classifier_two_sample_test(X_ref, X_same)["accuracy"] < 0.6


def test_importance_weights_track_density_ratio():
    rng = np.random.default_rng(0)
    X_ref = rng.normal(0, 1, size=(3000, 1))
    X_cur = rng.normal(1, 1, size=(3000, 1))
    xq = np.array([[-1.0], [0.0], [1.0], [2.0]])
    w = drift.importance_weights(X_ref, X_cur, xq)
    true = np.exp(xq[:, 0] - 0.5)  # N(1,1)/N(0,1) = exp(x - 1/2)
    assert np.all(np.diff(w) > 0) and np.allclose(np.log(w), np.log(true), atol=0.35)
