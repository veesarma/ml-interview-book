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
    return float(np.minimum(p, q).sum())


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """max(0, p - q) normalised; p, q: (V,) -> (V,)."""
    r = np.maximum(p - q, 0.0)  # (V,)
    s = r.sum()
    return r / s if s > 0 else p.copy()


def expected_tokens_per_call(alpha: float, k: int) -> float:
    """(1 - alpha^{k+1}) / (1 - alpha): expected accepted + 1 bonus token."""
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speculative_step(
    prev: int, P: np.ndarray, Q: np.ndarray, k: int, rng: np.random.Generator
) -> list[int]:
    """One speculative round: draft k tokens with Q, verify with P, return 1..k+1 tokens.

    P, Q: (V, V) row-stochastic bigram matrices; ``prev`` is the current token.
    Every emitted token is distributed exactly as the target would have sampled it.
    """
    draft: list[int] = []
    cur = prev
    for _ in range(k):  # draft phase: k sequential cheap samples
        cur = int(rng.choice(Q.shape[1], p=Q[cur]))
        draft.append(cur)
    # verify phase: the target scores all k+1 positions in ONE (parallel) call
    contexts = [prev] + draft[:-1]  # token preceding each drafted token
    out: list[int] = []
    for x, ctx in zip(draft, contexts):
        p_row, q_row = P[ctx], Q[ctx]  # (V,), (V,)
        if rng.random() < min(1.0, p_row[x] / q_row[x]):
            out.append(x)
        else:
            out.append(int(rng.choice(P.shape[1], p=residual_distribution(p_row, q_row))))
            return out  # everything after a rejection is discarded
    # all k accepted: a free extra token from the target's k+1-th distribution
    out.append(int(rng.choice(P.shape[1], p=P[draft[-1]])))
    return out


def make_toy_models(V: int, temperature_draft: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random bigram target P (V, V) and a draft Q = P smoothed toward uniform."""
    logits = rng.normal(size=(V, V))  # (V, V)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)  # (V, V) rows sum to 1
    Q = np.exp(logits / temperature_draft)
    Q /= Q.sum(axis=1, keepdims=True)
    return P, Q


def generate(P: np.ndarray, Q: np.ndarray, start: int, n_tokens: int, k: int, rng: np.random.Generator) -> tuple[list[int], int]:
    """Generate ``n_tokens`` with speculative rounds; return (tokens, number_of_target_calls)."""
    tokens: list[int] = []
    calls = 0
    cur = start
    while len(tokens) < n_tokens:
        new = speculative_step(cur, P, Q, k, rng)
        calls += 1
        tokens.extend(new)
        cur = new[-1]
    return tokens[:n_tokens], calls
