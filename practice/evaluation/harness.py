# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/evaluation/harness.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k harness -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py evaluation/harness --force

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
PerExampleMetric = Callable[[object, object], float]

@dataclass
class MetricRegistry:
    """Name -> per-example metric. Register your own with .add(name, fn)."""
    metrics: dict[str, PerExampleMetric] = field(default_factory=dict)

    def add(self, name: str, fn: PerExampleMetric) -> None:
        raise NotImplementedError('TODO: implement add (see the reference in src/mlbook)')

    def get(self, name: str) -> PerExampleMetric:
        raise NotImplementedError('TODO: implement get (see the reference in src/mlbook)')

def default_registry() -> MetricRegistry:
    raise NotImplementedError('TODO: implement default_registry (see the reference in src/mlbook)')

def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float]=np.mean, n_boot: int=1000, alpha: float=0.05, seed: int=0) -> tuple[float, float, float]:
    """Percentile bootstrap CI of stat(values). values (N,) -> (point, lo, hi)."""
    raise NotImplementedError('TODO: implement bootstrap_ci (see the reference in src/mlbook)')

def paired_bootstrap_test(a: np.ndarray, b: np.ndarray, n_boot: int=2000, alpha: float=0.05, seed: int=0) -> dict[str, float]:
    """Compare two systems scored on the SAME examples. a, b (N,).
    Returns the mean difference, its CI, and a two-sided bootstrap p-value."""
    raise NotImplementedError('TODO: implement paired_bootstrap_test (see the reference in src/mlbook)')

def clustered_standard_error(values: np.ndarray, clusters: np.ndarray) -> float:
    """SE of the mean when examples are grouped. values (N,), clusters (N,) ids.
    SE^2 = (1/N^2) sum_c (sum_{i in c} (x_i - x_bar))^2 * C/(C-1)."""
    raise NotImplementedError('TODO: implement clustered_standard_error (see the reference in src/mlbook)')

@dataclass
class EvalReport:
    n: int
    rows: list[dict]

    def to_markdown(self) -> str:
        raise NotImplementedError('TODO: implement to_markdown (see the reference in src/mlbook)')

class EvalSuite:
    """Run named per-example metrics over aligned predictions/references with CIs."""

    def __init__(self, registry: MetricRegistry | None=None, n_boot: int=1000, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def per_example(self, name: str, predictions: list, references: list) -> np.ndarray:
        raise NotImplementedError('TODO: implement per_example (see the reference in src/mlbook)')

    def run(self, predictions: list, references: list, metric_names: list[str]) -> EvalReport:
        raise NotImplementedError('TODO: implement run (see the reference in src/mlbook)')

def bootstrap_corpus_metric(y_true: np.ndarray, scores: np.ndarray, metric: str='roc_auc', n_boot: int=500, seed: int=0) -> tuple[float, float, float]:
    raise NotImplementedError('TODO: implement bootstrap_corpus_metric (see the reference in src/mlbook)')

def ece_with_ci(probs: np.ndarray, labels: np.ndarray, n_bins: int=15, n_boot: int=300, seed: int=0) -> tuple[float, float, float]:
    """Bootstrapped ECE (cross-link to reliability.calibration)."""
    raise NotImplementedError('TODO: implement ece_with_ci (see the reference in src/mlbook)')
