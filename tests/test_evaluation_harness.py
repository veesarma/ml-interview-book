import numpy as np

from mlbook.evaluation import harness as hz


def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(0)
    vals = rng.binomial(1, 0.7, size=400).astype(float)
    mean, lo, hi = hz.bootstrap_ci(vals, n_boot=500)
    assert lo < mean < hi
    se = np.sqrt(mean * (1 - mean) / 400)
    assert 0.8 * 3.92 * se < hi - lo < 1.2 * 3.92 * se  # close to the normal-approx width


def test_paired_bootstrap_test_detects_and_ignores_differences():
    rng = np.random.default_rng(0)
    base = rng.binomial(1, 0.6, size=300).astype(float)
    better = base.copy(); flip = rng.random(300) < 0.15; better[flip] = 1.0
    out = hz.paired_bootstrap_test(better, base, n_boot=500)
    assert out["diff"] > 0 and out["ci_lo"] > 0 and out["p_value"] < 0.05
    same = hz.paired_bootstrap_test(base, base, n_boot=200)
    assert same["diff"] == 0.0 and same["p_value"] >= 0.99


def test_clustered_standard_error_larger_than_iid_when_clusters_correlated():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(20), 10)
    cluster_effect = rng.normal(size=20)[clusters]
    vals = cluster_effect + 0.1 * rng.normal(size=200)
    se_iid = vals.std(ddof=1) / np.sqrt(200)
    assert hz.clustered_standard_error(vals, clusters) > 2 * se_iid


def test_eval_suite_runs_and_reports():
    suite = hz.EvalSuite(n_boot=100)
    preds = ["paris", "london", "rome", {"n": 4, "c": 2}]
    refs = ["Paris", "Berlin", "Rome", None]
    rep = suite.run(preds[:3], refs[:3], ["exact_match"])
    assert rep.n == 3 and np.isclose(rep.rows[0]["mean"], 2 / 3)
    assert "exact_match" in rep.to_markdown()
    p1 = suite.per_example("pass@1", [preds[3]], [refs[3]])
    assert np.isclose(p1[0], 0.5)


def test_metric_registry_custom_and_corpus_bootstrap():
    reg = hz.MetricRegistry(); reg.add("sq", lambda p, r: (p - r) ** 2)
    assert reg.get("sq")(3, 1) == 4
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 200); s = y + rng.normal(scale=0.8, size=200)
    auc, lo, hi = hz.bootstrap_corpus_metric(y, s, "roc_auc", n_boot=100)
    assert lo <= auc <= hi and auc > 0.7
    p = np.stack([1 - rng.random(200), np.zeros(200)], axis=1); p[:, 1] = 1 - p[:, 0]
    e, elo, ehi = hz.ece_with_ci(p, (p[:, 1] > 0.5).astype(int), n_boot=20)
    assert elo <= e <= ehi
