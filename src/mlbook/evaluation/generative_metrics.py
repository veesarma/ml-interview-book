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
    w, V = np.linalg.eigh(S)  # (d,), (d, d)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)[None, :]) @ V.T  # (d, d)


def frechet_distance(mu1: np.ndarray, S1: np.ndarray, mu2: np.ndarray, S2: np.ndarray) -> float:
    """Squared Fréchet distance between N(mu1, S1) and N(mu2, S2). mu (d,), S (d, d)."""
    diff = mu1 - mu2  # (d,)
    r1 = _psd_sqrt(S1)  # (d, d)
    inner = r1 @ S2 @ r1  # (d, d) symmetric PSD
    tr_cov_mean = np.trace(_psd_sqrt(inner))
    return float(diff @ diff + np.trace(S1) + np.trace(S2) - 2.0 * tr_cov_mean)


def fid_from_features(F1: np.ndarray, F2: np.ndarray) -> float:
    """FID from two feature matrices F1 (N1, d), F2 (N2, d)."""
    mu1, mu2 = F1.mean(axis=0), F2.mean(axis=0)  # (d,), (d,)
    S1 = np.cov(F1, rowvar=False)  # (d, d)
    S2 = np.cov(F2, rowvar=False)  # (d, d)
    return frechet_distance(mu1, S1, mu2, S2)


def inception_score(probs: np.ndarray, eps: float = 1e-12) -> float:
    """IS = exp( E_x KL(p(y|x) || p(y)) ), probs (N, K) rows summing to 1."""
    p_y = probs.mean(axis=0, keepdims=True)  # (1, K) marginal
    kl = np.sum(probs * (np.log(probs + eps) - np.log(p_y + eps)), axis=1)  # (N,)
    return float(np.exp(kl.mean()))


def clip_score(image_emb: np.ndarray, text_emb: np.ndarray, w: float = 2.5) -> float:
    """CLIPScore = w * mean_i max(cos(v_i, t_i), 0). image_emb (N, d), text_emb (N, d)."""
    v = image_emb / np.maximum(np.linalg.norm(image_emb, axis=1, keepdims=True), 1e-12)  # (N, d)
    t = text_emb / np.maximum(np.linalg.norm(text_emb, axis=1, keepdims=True), 1e-12)  # (N, d)
    cos = np.sum(v * t, axis=1)  # (N,)
    return float(w * np.mean(np.maximum(cos, 0.0)))
