"""Programmatic weak supervision (Snorkel-style): combine noisy labelling functions (LFs).

The label matrix is ``L ∈ {−1, 0, …, K−1}^{N×M}``: N examples, M labelling functions, −1 = abstain.
Two combiners:

* ``majority_vote``: count votes, ignore abstains.
* ``NaiveLabelModel``: a Dawid-Skene-style generative model: each LF j has a class prior-independent
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
    N, M = L.shape
    counts = np.zeros((N, n_classes))                             # (N, K)
    for k in range(n_classes):
        counts[:, k] = (L == k).sum(axis=1)                       # (N,)
    total = counts.sum(axis=1, keepdims=True)                     # (N, 1)
    probs = np.where(total > 0, counts / np.maximum(total, 1), 1.0 / n_classes)  # (N, K)
    return probs


class NaiveLabelModel:
    """Conditionally-independent generative label model fit by EM."""

    def __init__(self, n_classes: int, n_iter: int = 50, smoothing: float = 1.0) -> None:
        self.K = n_classes
        self.n_iter = n_iter
        self.smoothing = smoothing
        self.accuracy: np.ndarray | None = None                    # (M,) P(L_j = y | L_j ≠ −1)
        self.prior: np.ndarray | None = None                       # (K,)

    def _log_likelihood_matrix(self, L: np.ndarray) -> np.ndarray:
        """log P(L_i· | y_i = k) for every i, k, summing over LFs.  Returns (N, K)."""
        N, M = L.shape
        assert self.accuracy is not None
        acc = self.accuracy                                       # (M,)
        wrong = (1.0 - acc) / max(self.K - 1, 1)                  # (M,) mass spread over the K−1 wrong classes
        log_lik = np.zeros((N, self.K))                           # (N, K)
        for k in range(self.K):
            votes_k = (L == k)                                    # (N, M) LF j voted class k
            votes_other = (L != k) & (L != -1)                    # (N, M) LF j voted a different class
            log_lik[:, k] = votes_k @ np.log(acc) + votes_other @ np.log(wrong)  # (N,)
        return log_lik

    def fit(self, L: np.ndarray) -> "NaiveLabelModel":
        """EM from a majority-vote initialisation.  L: (N, M)."""
        N, M = L.shape
        q = majority_vote(L, self.K)                              # (N, K) initial responsibilities
        for _ in range(self.n_iter):
            # ---- M-step -------------------------------------------------------------
            self.prior = q.mean(axis=0)                           # (K,)
            correct = np.zeros(M)                                 # (M,) expected # correct votes
            for k in range(self.K):
                correct += ((L == k) * q[:, k][:, None]).sum(axis=0)  # (M,)
            voted = (L != -1).sum(axis=0)                         # (M,)
            self.accuracy = (correct + self.smoothing) / (voted + 2.0 * self.smoothing)  # (M,) Laplace-smoothed
            self.accuracy = np.clip(self.accuracy, 1e-3, 1 - 1e-3)
            # ---- E-step -------------------------------------------------------------
            log_post = np.log(self.prior)[None, :] + self._log_likelihood_matrix(L)  # (N, K)
            log_post -= log_post.max(axis=1, keepdims=True)       # (N, K) stabilise
            q = np.exp(log_post)
            q /= q.sum(axis=1, keepdims=True)                     # (N, K)
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """Posterior P(y_i = k | L_i·).  L: (N, M) → (N, K)."""
        assert self.prior is not None
        log_post = np.log(self.prior)[None, :] + self._log_likelihood_matrix(L)  # (N, K)
        log_post -= log_post.max(axis=1, keepdims=True)
        q = np.exp(log_post)
        return q / q.sum(axis=1, keepdims=True)


def lf_agreement_matrix(L: np.ndarray) -> np.ndarray:
    """Pairwise agreement rate between LFs on examples where both vote.  L: (N, M) → (M, M)."""
    N, M = L.shape
    A = np.eye(M)                                                 # (M, M)
    for i in range(M):
        for j in range(i + 1, M):
            both = (L[:, i] != -1) & (L[:, j] != -1)              # (N,)
            if both.sum() > 0:
                A[i, j] = A[j, i] = (L[both, i] == L[both, j]).mean()
    return A


def lf_coverage(L: np.ndarray) -> np.ndarray:
    """Fraction of examples each LF labels.  L: (N, M) → (M,)."""
    return (L != -1).mean(axis=0)
