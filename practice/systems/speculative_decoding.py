# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/speculative_decoding.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k speculative_decoding -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/speculative_decoding --force

"""Speculative sampling (Leviathan et al. 2023; Chen et al. 2023) on toy bigram models.

Target p(. | prev) and draft q(. | prev) are rows of (V, V) matrices. For one
position, draft x ~ q. Accept with probability min(1, p(x)/q(x)); on rejection
resample from the residual  r(x') = max(0, p(x') - q(x')) / sum_x max(0, p(x) - q(x)).

Why the output is exactly p:

    P(output = x) = q(x) min(1, p(x)/q(x)) + (1 - alpha) r(x)
                  = min(q(x), p(x)) + max(0, p(x) - q(x))       [since 1 - alpha = sum max(0, p - q)]
                  = p(x).

where alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q) is the acceptance probability.
With K drafted tokens the expected number produced per target call is
(1 - alpha^{K+1}) / (1 - alpha) (i.i.d. acceptance approximation).
"""
from __future__ import annotations
import numpy as np

def acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    """alpha = sum_x min(p, q) = 1 - TV(p, q) for one position."""
    raise NotImplementedError('TODO: implement acceptance_rate (see the reference in src/mlbook)')

def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """max(0, p - q) normalised; p, q: (V,) -> (V,)."""
    raise NotImplementedError('TODO: implement residual_distribution (see the reference in src/mlbook)')

def expected_tokens_per_call(alpha: float, k: int) -> float:
    """(1 - alpha^{k+1}) / (1 - alpha): expected accepted + 1 bonus token."""
    raise NotImplementedError('TODO: implement expected_tokens_per_call (see the reference in src/mlbook)')

def speculative_step(prev: int, P: np.ndarray, Q: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """One speculative round: draft k tokens with Q, verify with P, return 1..k+1 tokens.

    P, Q: (V, V) row-stochastic bigram matrices; ``prev`` is the current token.
    Every emitted token is distributed exactly as the target would have sampled it.
    """
    raise NotImplementedError('TODO: implement speculative_step (see the reference in src/mlbook)')

def make_toy_models(V: int, temperature_draft: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random bigram target P (V, V) and a draft Q = P smoothed toward uniform."""
    raise NotImplementedError('TODO: implement make_toy_models (see the reference in src/mlbook)')

def generate(P: np.ndarray, Q: np.ndarray, start: int, n_tokens: int, k: int, rng: np.random.Generator) -> tuple[list[int], int]:
    """Generate ``n_tokens`` with speculative rounds; return (tokens, number_of_target_calls)."""
    raise NotImplementedError('TODO: implement generate (see the reference in src/mlbook)')
