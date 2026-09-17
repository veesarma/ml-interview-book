# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/scaling_laws.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k scaling_laws -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/scaling_laws --force

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
CHINCHILLA_PAPER_FIT = ChinchillaParams(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
CHINCHILLA_REFIT = ChinchillaParams(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)
CHINCHILLA_FIT = CHINCHILLA_REFIT

def chinchilla_loss(N: np.ndarray, D: np.ndarray, p: ChinchillaParams=CHINCHILLA_FIT) -> np.ndarray:
    """Predicted loss for parameters ``N`` and tokens ``D`` (broadcast, any shape)."""
    raise NotImplementedError('TODO: implement chinchilla_loss (see the reference in src/mlbook)')

def training_flops(N: float, D: float) -> float:
    """C ≈ 6 N D: 2 FLOPs/param/token forward, 4 backward (dense Transformer)."""
    raise NotImplementedError('TODO: implement training_flops (see the reference in src/mlbook)')

def compute_optimal(C: float, p: ChinchillaParams=CHINCHILLA_FIT) -> tuple[float, float]:
    """Compute-optimal (N*, D*) for a FLOP budget ``C`` under ``L(N, D)`` and ``C = 6ND``.

    Returns:
        (N_star, D_star) in parameters and tokens.
    """
    raise NotImplementedError('TODO: implement compute_optimal (see the reference in src/mlbook)')

def tokens_per_parameter(C: float, p: ChinchillaParams=CHINCHILLA_FIT) -> float:
    """D*/N* at budget ``C`` (≈ 20 for the Chinchilla fit around 1e21–1e24 FLOPs)."""
    raise NotImplementedError('TODO: implement tokens_per_parameter (see the reference in src/mlbook)')

def loss_at_fixed_compute(C: float, N: np.ndarray, p: ChinchillaParams=CHINCHILLA_FIT) -> np.ndarray:
    """Loss along an iso-FLOP curve: D = C / (6N) for each ``N`` (the 'IsoFLOP' view)."""
    raise NotImplementedError('TODO: implement loss_at_fixed_compute (see the reference in src/mlbook)')

def fit_scaling_law(N: np.ndarray, D: np.ndarray, L: np.ndarray, alpha_grid=None, beta_grid=None) -> ChinchillaParams:
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
    raise NotImplementedError('TODO: implement fit_scaling_law (see the reference in src/mlbook)')

def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Kaplan-style single-variable fit ``y = c x^-k`` by linear regression in log-log.

    Returns:
        (c, k).
    """
    raise NotImplementedError('TODO: implement fit_power_law (see the reference in src/mlbook)')

def effective_data_with_repetition(D: float, U: float, R_star: float=15.0) -> float:
    """Data-constrained scaling (Muennighoff et al. 2023): effective unique tokens.

    With ``U`` unique tokens trained for ``D/U`` epochs, the ``R_D = D/U - 1``
    repeated passes are worth ``D' = U + U R* (1 - exp(-R_D / R*))`` unique tokens;
    ``R* ≈ 15`` is their fitted decay constant.  Up to ~4 epochs, ``D' ≈ D``.
    """
    raise NotImplementedError('TODO: implement effective_data_with_repetition (see the reference in src/mlbook)')

def inference_flops_per_token(N: float) -> float:
    """≈ 2N FLOPs per generated token (one forward pass, ignoring attention)."""
    raise NotImplementedError('TODO: implement inference_flops_per_token (see the reference in src/mlbook)')

def lifetime_cost_flops(N: float, D_train: float, tokens_served: float) -> float:
    """Training plus inference FLOPs, the quantity inference-aware scaling minimises."""
    raise NotImplementedError('TODO: implement lifetime_cost_flops (see the reference in src/mlbook)')
