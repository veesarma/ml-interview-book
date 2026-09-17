"""LLM evaluation primitives (NumPy): exact match, unbiased pass@k, Bradley-Terry
leaderboard fitting, Elo updates, and judge-validation helpers (position
consistency, agreement with humans, Cohen's kappa).
"""
from __future__ import annotations

import re
import string

import numpy as np


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lower-case, strip punctuation/articles/extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k = 1 - C(n-c, k) / C(n, k), computed as a stable product
    (Chen et al. 2021, Codex). n samples, c correct, k <= n."""
    if n - c < k:
        return 1.0
    # C(n-c,k)/C(n,k) = prod_{i=n-c+1..n} (i-k)/i  -> 1 - prod(1 - k/i) for i in n-c+1..n
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def bradley_terry(wins: np.ndarray, n_iters: int = 200, eps: float = 1e-9) -> np.ndarray:
    """Fit Bradley-Terry strengths by the Zermelo / MM iteration.

    wins (M, M): wins[i, j] = number of times model i beat j. Model:
        P(i beats j) = p_i / (p_i + p_j).
    Update: p_i <- W_i / sum_{j != i} (n_ij / (p_i + p_j)),  then renormalise.
    Returns log-strengths (M,) centred at 0 (Elo-like scale if multiplied by 400/ln10).
    """
    M = wins.shape[0]
    n = wins + wins.T  # (M, M) games played between i and j
    W = wins.sum(axis=1)  # (M,) total wins of i
    p = np.ones(M)  # (M,)
    for _ in range(n_iters):
        denom = np.zeros(M)  # (M,)
        for i in range(M):
            mask = np.arange(M) != i
            denom[i] = np.sum(n[i, mask] / (p[i] + p[mask]))
        p_new = (W + eps) / (denom + eps)  # (M,)
        p_new /= np.exp(np.mean(np.log(p_new)))  # geometric-mean normalisation
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new
            break
        p = p_new
    return np.log(p)


def elo_update(r_a: float, r_b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    """One Elo step. score_a in {1, 0.5, 0}. E_a = 1 / (1 + 10^((r_b - r_a)/400))."""
    e_a = 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))
    delta = k * (score_a - e_a)
    return r_a + delta, r_b - delta


def position_consistency(verdict_ab: np.ndarray, verdict_ba: np.ndarray) -> float:
    """Judge position bias check. verdict_ab[i] in {"A","B","tie"} when the pair is shown
    as (A,B); verdict_ba[i] when shown swapped. Consistent iff the same underlying
    response wins both times. Returns the fraction of consistent judgments."""
    swap = {"A": "B", "B": "A", "tie": "tie"}
    consistent = [a == swap[b] for a, b in zip(verdict_ab, verdict_ba)]
    return float(np.mean(consistent))


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement corrected for chance: (p_o - p_e) / (1 - p_e)."""
    labels = np.unique(np.concatenate([a, b]))
    p_o = float(np.mean(a == b))
    p_e = float(sum(np.mean(a == l) * np.mean(b == l) for l in labels))
    return (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0


def judge_agreement(judge: np.ndarray, human: np.ndarray) -> dict[str, float]:
    """Raw agreement and Cohen's kappa between a judge's labels and human labels."""
    return {"agreement": float(np.mean(judge == human)), "kappa": cohens_kappa(judge, human)}
