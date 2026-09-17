# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/evaluation/generative_metrics.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k generative_metrics -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py evaluation/generative_metrics --force

"""Generative-model metrics (NumPy): Fréchet distance between Gaussians (FID),
Inception Score, CLIPScore.

FID^2 = ||mu_1 - mu_2||^2 + tr(S_1 + S_2 - 2 (S_1 S_2)^{1/2}).
We compute tr((S_1 S_2)^{1/2}) as tr((S_1^{1/2} S_2 S_1^{1/2})^{1/2}), which is the
trace of the square root of a symmetric PSD matrix and therefore needs only eigh.
"""
from __future__ import annotations
import numpy as np

def _psd_sqrt(S: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition. (d, d) -> (d, d)."""
    raise NotImplementedError('TODO: implement _psd_sqrt (see the reference in src/mlbook)')

def frechet_distance(mu1: np.ndarray, S1: np.ndarray, mu2: np.ndarray, S2: np.ndarray) -> float:
    """Squared Fréchet distance between N(mu1, S1) and N(mu2, S2). mu (d,), S (d, d)."""
    raise NotImplementedError('TODO: implement frechet_distance (see the reference in src/mlbook)')

def fid_from_features(F1: np.ndarray, F2: np.ndarray) -> float:
    """FID from two feature matrices F1 (N1, d), F2 (N2, d)."""
    raise NotImplementedError('TODO: implement fid_from_features (see the reference in src/mlbook)')

def inception_score(probs: np.ndarray, eps: float=1e-12) -> float:
    """IS = exp( E_x KL(p(y|x) || p(y)) ), probs (N, K) rows summing to 1."""
    raise NotImplementedError('TODO: implement inception_score (see the reference in src/mlbook)')

def clip_score(image_emb: np.ndarray, text_emb: np.ndarray, w: float=2.5) -> float:
    """CLIPScore = w * mean_i max(cos(v_i, t_i), 0). image_emb (N, d), text_emb (N, d)."""
    raise NotImplementedError('TODO: implement clip_score (see the reference in src/mlbook)')
