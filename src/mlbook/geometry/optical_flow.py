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
    Ix = np.zeros_like(img)  # (H, W)
    Iy = np.zeros_like(img)  # (H, W)
    Ix[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    Iy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    return Ix, Iy


def lucas_kanade(img1: np.ndarray, img2: np.ndarray, points: np.ndarray, window: int = 7, iters: int = 5, smooth_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Iterative LK flow at sparse points (each an (y, x) pixel location).

    Each iteration warps ``img2`` by the current estimate (bilinear), recomputes
    ``I_t = I₂(x + d) − I₁(x)`` and solves the 2×2 system for the update.

    Args:
        img1, img2: (H, W) grayscale.  points: (N, 2) as (y, x).
    Returns:
        flow (N, 2) as (dy, dx);  min_eigenvalue (N,) of the structure tensor (trackability).
    """
    a = gaussian_blur(img1, smooth_sigma)  # (H, W)
    b = gaussian_blur(img2, smooth_sigma)  # (H, W)
    Ix, Iy = image_gradients(a)  # (H, W) each
    r = window // 2
    dy_off, dx_off = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")  # (w, w) each
    flow = np.zeros((len(points), 2))  # (N, 2)
    min_eig = np.zeros(len(points))  # (N,)
    for n, (py, px) in enumerate(points):
        ys = (py + dy_off).ravel()  # (w²,) window rows
        xs = (px + dx_off).ravel()  # (w²,)
        gx = bilinear_sample(Ix, ys, xs)  # (w²,)
        gy = bilinear_sample(Iy, ys, xs)  # (w²,)
        G = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])  # (2, 2) structure tensor
        min_eig[n] = np.linalg.eigvalsh(G)[0]
        if min_eig[n] < 1e-6:
            continue  # aperture problem: flow not determined here
        I1 = bilinear_sample(a, ys, xs)  # (w²,)
        d = np.zeros(2)  # (dy, dx)
        for _ in range(iters):
            I2 = bilinear_sample(b, ys + d[0], xs + d[1])  # (w²,) warped second image
            It = I2 - I1  # (w²,) temporal difference under current estimate
            rhs = -np.array([np.sum(gx * It), np.sum(gy * It)])  # (2,) as (x, y)
            step = np.linalg.solve(G, rhs)  # (2,) = (du, dv)
            d = d + np.array([step[1], step[0]])  # accumulate as (dy, dx)
        flow[n] = d
    return flow, min_eig
