# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/optical_flow.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k optical_flow -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/optical_flow --force

"""Lucas–Kanade optical flow (NumPy): brightness constancy + local constant flow.

Brightness constancy ``I(x + u, y + v, t + 1) = I(x, y, t)``, linearised:
    I_x u + I_y v + I_t = 0            (one equation, two unknowns per pixel)
Lucas–Kanade assumes ``(u, v)`` is constant in a window ``W`` and solves the normal equations
    [Σ I_x²   Σ I_x I_y] [u]   = −[Σ I_x I_t]
    [Σ I_x I_y  Σ I_y² ] [v]      [Σ I_y I_t]
whose matrix is the structure tensor — invertible only at corners (both eigenvalues large).
"""
from __future__ import annotations
import numpy as np
from mlbook.vision.image_ops import bilinear_sample, gaussian_blur

def image_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences ``I_x, I_y`` of an (H, W) image, each (H, W)."""
    raise NotImplementedError('TODO: implement image_gradients (see the reference in src/mlbook)')

def lucas_kanade(img1: np.ndarray, img2: np.ndarray, points: np.ndarray, window: int=7, iters: int=5, smooth_sigma: float=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Iterative LK flow at sparse points (each an (y, x) pixel location).

    Each iteration warps ``img2`` by the current estimate (bilinear), recomputes
    ``I_t = I₂(x + d) − I₁(x)`` and solves the 2×2 system for the update.

    Args:
        img1, img2: (H, W) grayscale.  points: (N, 2) as (y, x).
    Returns:
        flow (N, 2) as (dy, dx);  min_eigenvalue (N,) of the structure tensor (trackability).
    """
    raise NotImplementedError('TODO: implement lucas_kanade (see the reference in src/mlbook)')
