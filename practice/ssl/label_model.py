# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/ssl/label_model.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k label_model -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py ssl/label_model --force

"""Programmatic weak supervision (Snorkel-style): combine noisy labelling functions (LFs).

The label matrix is ``L ∈ {−1, 0, …, K−1}^{N×M}``: N examples, M labelling functions, −1 = abstain.
Two combiners:

* ``majority_vote`` — count votes, ignore abstains.
* ``NaiveLabelModel`` — a Dawid–Skene-style generative model: each LF j has a class prior-independent
  accuracy ``a_j`` and a propensity to vote ``β_j``; conditioned on the true label y the LF votes are
  independent (the *data programming* assumption).  Fit by EM with no ground truth:
      E-step  q(y_i = k) ∝ π_k Π_j P(L_ij | y = k)
      M-step  a_j = Σ_i Σ_k q_ik 1[L_ij = k] / Σ_i 1[L_ij ≠ −1],   π_k = mean_i q_ik.
"""
from __future__ import annotations
import numpy as np

def majority_vote(L: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-example vote histogram over non-abstaining LFs, normalised to probabilities.

    Args:
        L: (N, M) ints in {−1, 0, …, K−1}.
    Returns:
        probs: (N, K); uniform for examples where every LF abstains.
    """
    raise NotImplementedError('TODO: implement majority_vote (see the reference in src/mlbook)')

class NaiveLabelModel:
    """Conditionally-independent generative label model fit by EM."""

    def __init__(self, n_classes: int, n_iter: int=50, smoothing: float=1.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _log_likelihood_matrix(self, L: np.ndarray) -> np.ndarray:
        """log P(L_i· | y_i = k) for every i, k, summing over LFs.  Returns (N, K)."""
        raise NotImplementedError('TODO: implement _log_likelihood_matrix (see the reference in src/mlbook)')

    def fit(self, L: np.ndarray) -> 'NaiveLabelModel':
        """EM from a majority-vote initialisation.  L: (N, M)."""
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Posterior P(y_i = k | L_i·).  L: (N, M) → (N, K)."""
        raise NotImplementedError('TODO: implement predict_proba (see the reference in src/mlbook)')

def lf_agreement_matrix(L: np.ndarray) -> np.ndarray:
    """Pairwise agreement rate between LFs on examples where both vote.  L: (N, M) → (M, M)."""
    raise NotImplementedError('TODO: implement lf_agreement_matrix (see the reference in src/mlbook)')

def lf_coverage(L: np.ndarray) -> np.ndarray:
    """Fraction of examples each LF labels.  L: (N, M) → (M,)."""
    raise NotImplementedError('TODO: implement lf_coverage (see the reference in src/mlbook)')
