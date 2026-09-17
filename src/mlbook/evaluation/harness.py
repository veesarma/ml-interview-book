"""A small evaluation harness: metric registry, bootstrap confidence intervals,
paired comparison test, and a suite runner that turns (predictions, references)
into a report with error bars.

Statistics used
    bootstrap CI     resample examples with replacement B times, take the
                     (alpha/2, 1 - alpha/2) percentiles of the statistic
    paired bootstrap resample *the same indices* for both systems so the CI of
                     the difference reflects question-level pairing; the p-value is
                     the fraction of resamples where the sign flips (two-sided)
    clustered SE     when examples come in clusters (e.g. several questions per
                     document) resample clusters, not examples
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mlbook.evaluation.classification_metrics import average_precision, roc_auc
from mlbook.evaluation.llm_metrics import exact_match, pass_at_k
from mlbook.evaluation.ranking_metrics import ndcg_at_k
from mlbook.reliability.calibration import expected_calibration_error

# A per-example metric maps (prediction, reference) -> float so it can be bootstrapped.
PerExampleMetric = Callable[[object, object], float]


@dataclass
class MetricRegistry:
    """Name -> per-example metric. Register your own with .add(name, fn)."""

    metrics: dict[str, PerExampleMetric] = field(default_factory=dict)

    def add(self, name: str, fn: PerExampleMetric) -> None:
        self.metrics[name] = fn

    def get(self, name: str) -> PerExampleMetric:
        return self.metrics[name]


def default_registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.add("exact_match", lambda p, r: exact_match(str(p), str(r)))
    reg.add("ndcg@10", lambda p, r: ndcg_at_k(np.asarray(p), 10))  # p = ranked rels
    reg.add("pass@1", lambda p, r: pass_at_k(int(p["n"]), int(p["c"]), 1))
    reg.add("abs_error", lambda p, r: abs(float(p) - float(r)))
    return reg


def bootstrap_ci(
    values: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of stat(values). values (N,) -> (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    N = len(values)
    idx = rng.integers(0, N, size=(n_boot, N))  # (B, N) resampled indices
    boots = np.array([stat(values[row]) for row in idx])  # (B,)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(values)), float(lo), float(hi)


def paired_bootstrap_test(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Compare two systems scored on the SAME examples. a, b (N,).
    Returns the mean difference, its CI, and a two-sided bootstrap p-value."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    d = a - b  # (N,) per-example paired differences
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))  # (B, N)
    boots = d[idx].mean(axis=1)  # (B,)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    observed = float(d.mean())
    # two-sided: how often does the resampled mean cross zero on the other side?
    frac_opposite = float(np.mean(boots <= 0.0)) if observed > 0 else float(np.mean(boots >= 0.0))
    p_value = min(1.0, 2.0 * frac_opposite)
    return {"diff": observed, "ci_lo": float(lo), "ci_hi": float(hi), "p_value": p_value}


def clustered_standard_error(values: np.ndarray, clusters: np.ndarray) -> float:
    """SE of the mean when examples are grouped. values (N,), clusters (N,) ids.
    SE^2 = (1/N^2) sum_c (sum_{i in c} (x_i - x_bar))^2 * C/(C-1)."""
    xbar = values.mean()
    ids = np.unique(clusters)
    C = len(ids)
    sums = np.array([np.sum(values[clusters == c] - xbar) for c in ids])  # (C,)
    return float(np.sqrt(np.sum(sums**2) * C / max(C - 1, 1)) / len(values))


@dataclass
class EvalReport:
    n: int
    rows: list[dict]  # one per metric: name, mean, ci_lo, ci_hi

    def to_markdown(self) -> str:
        lines = [f"| metric | mean | 95% CI | n |", "|---|---|---|---|"]
        for r in self.rows:
            lines.append(f"| {r['name']} | {r['mean']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {self.n} |")
        return "\n".join(lines)


class EvalSuite:
    """Run named per-example metrics over aligned predictions/references with CIs."""

    def __init__(self, registry: MetricRegistry | None = None, n_boot: int = 1000, seed: int = 0) -> None:
        self.registry = registry or default_registry()
        self.n_boot = n_boot
        self.seed = seed

    def per_example(self, name: str, predictions: list, references: list) -> np.ndarray:
        fn = self.registry.get(name)
        return np.array([fn(p, r) for p, r in zip(predictions, references)], dtype=float)  # (N,)

    def run(self, predictions: list, references: list, metric_names: list[str]) -> EvalReport:
        assert len(predictions) == len(references)
        rows = []
        for name in metric_names:
            vals = self.per_example(name, predictions, references)  # (N,)
            mean, lo, hi = bootstrap_ci(vals, np.mean, self.n_boot, seed=self.seed)
            rows.append({"name": name, "mean": mean, "ci_lo": lo, "ci_hi": hi})
        return EvalReport(n=len(predictions), rows=rows)


# Corpus-level (non-decomposable) metrics that the harness can still bootstrap by
# resampling examples and recomputing the statistic on the resample.
def bootstrap_corpus_metric(
    y_true: np.ndarray, scores: np.ndarray, metric: str = "roc_auc", n_boot: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    fn = {"roc_auc": roc_auc, "average_precision": average_precision}[metric]
    rng = np.random.default_rng(seed)
    N = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)  # (N,)
        if y_true[idx].min() == y_true[idx].max():
            continue  # resample lost a class; skip
        boots.append(fn(y_true[idx], scores[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(fn(y_true, scores)), float(lo), float(hi)


def ece_with_ci(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, n_boot: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrapped ECE (cross-link to reliability.calibration)."""
    rng = np.random.default_rng(seed)
    N = len(labels)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots.append(expected_calibration_error(probs[idx], labels[idx], n_bins)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return expected_calibration_error(probs, labels, n_bins)[0], float(lo), float(hi)
