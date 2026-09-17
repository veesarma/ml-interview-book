"""Neural scaling laws: the Chinchilla parametric form, compute-optimal allocation,
and fitting a law to a handful of small runs (pure NumPy).

    L(N, D) = E + A / N^alpha + B / D^beta            (Hoffmann et al. 2022, approach 3)
    C ≈ 6 N D  training FLOPs for a dense Transformer (2 fwd + 4 bwd per param per token)

Minimising L subject to C = 6ND gives (derivation in the chapter)

    N*(C) = G (C/6)^a,  D*(C) = G^{-1} (C/6)^b,
    a = beta / (alpha + beta), b = alpha / (alpha + beta),
    G = (alpha A / (beta B))^{1 / (alpha + beta)}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChinchillaParams:
    """Parameters of L(N, D) = E + A N^-alpha + B D^-beta."""

    E: float
    A: float
    B: float
    alpha: float
    beta: float


# Fitted values as printed in Hoffmann et al. (2022), "approach 3" (their eq. 10).
# NOTE: with these exponents the closed-form optimum gives N* ∝ C^0.46 and ~50+
# tokens/parameter at 1e21 FLOPs, *not* the "20 tokens per parameter" rule that
# came from approaches 1 and 2 of the same paper.  Besiroglu et al. (2024,
# "Chinchilla Scaling: A replication attempt") re-fit the same data and obtained
# the second parameter set below, which is consistent with all three approaches
# and with the actual Chinchilla configuration (70B params, 1.4T tokens).
CHINCHILLA_PAPER_FIT = ChinchillaParams(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
CHINCHILLA_REFIT = ChinchillaParams(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)
CHINCHILLA_FIT = CHINCHILLA_REFIT  # default used throughout the chapter


def chinchilla_loss(N: np.ndarray, D: np.ndarray, p: ChinchillaParams = CHINCHILLA_FIT) -> np.ndarray:
    """Predicted loss for parameters ``N`` and tokens ``D`` (broadcast, any shape)."""
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    return p.E + p.A / N**p.alpha + p.B / D**p.beta


def training_flops(N: float, D: float) -> float:
    """C ≈ 6 N D: 2 FLOPs/param/token forward, 4 backward (dense Transformer)."""
    return 6.0 * N * D


def compute_optimal(C: float, p: ChinchillaParams = CHINCHILLA_FIT) -> tuple[float, float]:
    """Compute-optimal (N*, D*) for a FLOP budget ``C`` under ``L(N, D)`` and ``C = 6ND``.

    Returns:
        (N_star, D_star) in parameters and tokens.
    """
    a = p.beta / (p.alpha + p.beta)
    b = p.alpha / (p.alpha + p.beta)
    G = (p.alpha * p.A / (p.beta * p.B)) ** (1.0 / (p.alpha + p.beta))
    N_star = G * (C / 6.0) ** a
    D_star = (C / 6.0) ** b / G
    return N_star, D_star


def tokens_per_parameter(C: float, p: ChinchillaParams = CHINCHILLA_FIT) -> float:
    """D*/N* at budget ``C`` (≈ 20 for the Chinchilla fit around 1e21–1e24 FLOPs)."""
    N_star, D_star = compute_optimal(C, p)
    return D_star / N_star


def loss_at_fixed_compute(C: float, N: np.ndarray, p: ChinchillaParams = CHINCHILLA_FIT) -> np.ndarray:
    """Loss along an iso-FLOP curve: D = C / (6N) for each ``N`` (the 'IsoFLOP' view)."""
    N = np.asarray(N, dtype=float)
    D = C / (6.0 * N)
    return chinchilla_loss(N, D, p)


def fit_scaling_law(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, alpha_grid=None, beta_grid=None
) -> ChinchillaParams:
    """Fit ``E, A, B, alpha, beta`` to observed (N, D, L) triples from small runs.

    For fixed exponents the model is *linear* in ``(E, A, B)`` with features
    ``[1, N^-alpha, D^-beta]``, so we solve least squares in closed form and grid
    search the two exponents.  Robust, dependency-free, and good enough for the
    handful of points you get from a scaling sweep.

    Args:
        N, D, L: (n_runs,) arrays of parameter counts, token counts, final losses.
    Returns:
        the best ChinchillaParams (lowest squared error).
    """
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    alpha_grid = np.linspace(0.1, 0.9, 41) if alpha_grid is None else alpha_grid
    beta_grid = np.linspace(0.1, 0.9, 41) if beta_grid is None else beta_grid
    best: tuple[float, ChinchillaParams] | None = None
    for alpha in alpha_grid:
        for beta in beta_grid:
            X = np.stack([np.ones_like(N), N**-alpha, D**-beta], axis=1)  # (n_runs, 3)
            coef, *_ = np.linalg.lstsq(X, L, rcond=None)  # (3,) = [E, A, B]
            if np.any(coef < 0):  # E, A, B must be positive to be a scaling law
                continue
            err = float(np.sum((X @ coef - L) ** 2))
            if best is None or err < best[0]:
                best = (err, ChinchillaParams(coef[0], coef[1], coef[2], float(alpha), float(beta)))
    assert best is not None
    return best[1]


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Kaplan-style single-variable fit ``y = c x^-k`` by linear regression in log-log.

    Returns:
        (c, k).
    """
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    k, logc = np.polyfit(lx, ly, deg=1)  # slope, intercept
    return float(np.exp(logc)), float(-k)


def effective_data_with_repetition(D: float, U: float, R_star: float = 15.0) -> float:
    """Data-constrained scaling (Muennighoff et al. 2023): effective unique tokens.

    With ``U`` unique tokens trained for ``D/U`` epochs, the ``R_D = D/U - 1``
    repeated passes are worth ``D' = U + U R* (1 - exp(-R_D / R*))`` unique tokens;
    ``R* ≈ 15`` is their fitted decay constant.  Up to ~4 epochs, ``D' ≈ D``.
    """
    if D <= U:
        return D
    R_D = D / U - 1.0
    return U + U * R_star * (1.0 - np.exp(-R_D / R_star))


def inference_flops_per_token(N: float) -> float:
    """≈ 2N FLOPs per generated token (one forward pass, ignoring attention)."""
    return 2.0 * N


def lifetime_cost_flops(N: float, D_train: float, tokens_served: float) -> float:
    """Training plus inference FLOPs, the quantity inference-aware scaling minimises."""
    return training_flops(N, D_train) + inference_flops_per_token(N) * tokens_served
