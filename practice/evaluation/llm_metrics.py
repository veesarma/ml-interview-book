# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/evaluation/llm_metrics.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k llm_metrics -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py evaluation/llm_metrics --force

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
    raise NotImplementedError('TODO: implement normalize_answer (see the reference in src/mlbook)')

def exact_match(pred: str, ref: str) -> float:
    raise NotImplementedError('TODO: implement exact_match (see the reference in src/mlbook)')

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k = 1 - C(n-c, k) / C(n, k), computed as a stable product
    (Chen et al. 2021, Codex). n samples, c correct, k <= n."""
    raise NotImplementedError('TODO: implement pass_at_k (see the reference in src/mlbook)')

def bradley_terry(wins: np.ndarray, n_iters: int=200, eps: float=1e-09) -> np.ndarray:
    """Fit Bradley-Terry strengths by the Zermelo / MM iteration.

    wins (M, M): wins[i, j] = number of times model i beat j. Model:
        P(i beats j) = p_i / (p_i + p_j).
    Update: p_i <- W_i / sum_{j != i} (n_ij / (p_i + p_j)),  then renormalise.
    Returns log-strengths (M,) centred at 0 (Elo-like scale if multiplied by 400/ln10).
    """
    raise NotImplementedError('TODO: implement bradley_terry (see the reference in src/mlbook)')

def elo_update(r_a: float, r_b: float, score_a: float, k: float=32.0) -> tuple[float, float]:
    """One Elo step. score_a in {1, 0.5, 0}. E_a = 1 / (1 + 10^((r_b - r_a)/400))."""
    raise NotImplementedError('TODO: implement elo_update (see the reference in src/mlbook)')

def position_consistency(verdict_ab: np.ndarray, verdict_ba: np.ndarray) -> float:
    """Judge position bias check. verdict_ab[i] in {"A","B","tie"} when the pair is shown
    as (A,B); verdict_ba[i] when shown swapped. Consistent iff the same underlying
    response wins both times. Returns the fraction of consistent judgments."""
    raise NotImplementedError('TODO: implement position_consistency (see the reference in src/mlbook)')

def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement corrected for chance: (p_o - p_e) / (1 - p_e)."""
    raise NotImplementedError('TODO: implement cohens_kappa (see the reference in src/mlbook)')

def judge_agreement(judge: np.ndarray, human: np.ndarray) -> dict[str, float]:
    """Raw agreement and Cohen's kappa between a judge's labels and human labels."""
    raise NotImplementedError('TODO: implement judge_agreement (see the reference in src/mlbook)')
