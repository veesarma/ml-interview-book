import numpy as np

from mlbook.reliability import conformal as cp


def test_conformal_quantile_finite_sample_correction():
    scores = np.arange(1, 11, dtype=float)  # n = 10
    # alpha=0.1: ceil(11*0.9)/10 = 10/10 -> the max
    assert cp.conformal_quantile(scores, 0.1) == 10.0
    assert cp.conformal_quantile(scores, 0.5) == 6.0  # ceil(5.5)=6 -> 0.6 quantile 'higher' = 6
    assert np.isinf(cp.conformal_quantile(scores, 0.05))  # n too small


def test_split_conformal_classifier_coverage():
    rng = np.random.default_rng(0)
    K, n_cal, n_test = 5, 2000, 4000
    def sample(n):
        y = rng.integers(0, K, n)
        logits = rng.normal(size=(n, K)); logits[np.arange(n), y] += 1.5
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        return p, y
    pc, yc = sample(n_cal); pt, yt = sample(n_test)
    for alpha in (0.05, 0.2):
        c = cp.SplitConformalClassifier(alpha).calibrate(pc, yc)
        cov = cp.empirical_coverage_sets(c.predict_sets(pt), yt)
        assert 1 - alpha - 0.03 <= cov <= 1 - alpha + 0.04


def test_split_conformal_regressor_coverage():
    rng = np.random.default_rng(1)
    mu_cal, mu_test = rng.normal(size=1000), rng.normal(size=3000)
    y_cal = mu_cal + rng.standard_t(3, size=1000)
    y_test = mu_test + rng.standard_t(3, size=3000)
    r = cp.SplitConformalRegressor(alpha=0.1).calibrate(mu_cal, y_cal)
    lo, hi = r.predict_intervals(mu_test)
    cov = cp.empirical_coverage_intervals(lo, hi, y_test)
    assert 0.87 <= cov <= 0.94 and np.allclose(hi - lo, 2 * r.q_hat)
