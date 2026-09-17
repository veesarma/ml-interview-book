"""Linear Kalman filter and the constant-velocity model (NumPy).

State ``x ∈ R^n``, measurement ``z ∈ R^m``, linear-Gaussian model

    x_t = F x_{t−1} + w,   w ~ N(0, Q)          (motion)
    z_t = H x_t + v,       v ~ N(0, R)          (measurement)

Predict:   x̂⁻ = F x̂,            P⁻ = F P Fᵀ + Q
Update:    S = H P⁻ Hᵀ + R,      K = P⁻ Hᵀ S⁻¹,
           x̂ = x̂⁻ + K (z − H x̂⁻),   P = (I − K H) P⁻

``K`` is the ratio of prior uncertainty to total uncertainty in measurement space; the
innovation ``y = z − H x̂⁻`` with covariance ``S`` gives the Mahalanobis gate used by
DeepSORT and every 3D tracker.
"""

from __future__ import annotations

import numpy as np


def constant_velocity_model(n_pos: int, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``F`` (2n, 2n) and ``H`` (n, 2n) for state ``[p_1..p_n, v_1..v_n]`` with ``p += v·dt``."""
    F = np.eye(2 * n_pos)  # (2n, 2n)
    F[:n_pos, n_pos:] = dt * np.eye(n_pos)
    H = np.zeros((n_pos, 2 * n_pos))  # (n, 2n)
    H[:, :n_pos] = np.eye(n_pos)
    return F, H


class KalmanFilter:
    """Predict / update for a linear-Gaussian state-space model.

    Attributes: ``x`` (n,), ``P`` (n, n), ``F`` (n, n), ``H`` (m, n), ``Q`` (n, n), ``R`` (m, m).
    """

    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x = x0.astype(np.float64).copy()  # (n,)
        self.P = P0.astype(np.float64).copy()  # (n, n)

    def predict(self) -> np.ndarray:
        """``x̂⁻ = F x̂``, ``P⁻ = F P Fᵀ + Q``.  Returns the predicted state (n,)."""
        self.x = self.F @ self.x  # (n,)
        self.P = self.F @ self.P @ self.F.T + self.Q  # (n, n)
        return self.x

    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``y = z − H x̂⁻`` (m,) and ``S = H P⁻ Hᵀ + R`` (m, m)."""
        y = z - self.H @ self.x  # (m,)
        S = self.H @ self.P @ self.H.T + self.R  # (m, m)
        return y, S

    def mahalanobis(self, z: np.ndarray) -> float:
        """Squared Mahalanobis distance ``yᵀ S⁻¹ y`` of a measurement; χ²(m) distributed under the model."""
        y, S = self.innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z: np.ndarray) -> np.ndarray:
        """Kalman update with measurement ``z`` (m,).  Returns the posterior state (n,).

        ``K = P⁻ Hᵀ S⁻¹`` is computed with a solve (never an explicit inverse), and the
        covariance uses the Joseph form ``(I−KH) P (I−KH)ᵀ + K R Kᵀ`` for numerical symmetry.
        """
        y, S = self.innovation(z)
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T  # (n, m)   K = P Hᵀ S⁻¹
        self.x = self.x + K @ y  # (n,)
        I_KH = np.eye(self.P.shape[0]) - K @ self.H  # (n, n)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # (n, n)
        return self.x


def covariance_ellipse(P2: np.ndarray, n_std: float = 2.0, n_points: int = 64) -> np.ndarray:
    """Points on the ``n_std``-sigma ellipse of a 2×2 covariance (for figures).

    The ellipse is the level set ``{p : pᵀ P⁻¹ p = n_std²}``.  Writing ``P = V Λ Vᵀ`` (eigh),
    a unit circle scaled by ``n_std·√Λ`` and rotated by ``V`` traces exactly that set.

    Args:
        P2: (2, 2) symmetric positive semi-definite covariance.
        n_points: number of *distinct* angles; the returned polygon repeats the first point
            at the end so it closes, hence shape ``(n_points + 1, 2)``.  Angles are sampled
            with ``endpoint=False`` so that a power-of-two ``n_points`` lands exactly on the
            principal axes (a closed ``linspace`` would miss them and shrink the drawn axes).
    """
    vals, vecs = np.linalg.eigh(P2)  # (2,) ascending eigenvalues, (2, 2) eigenvectors in columns
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)  # (n_points,)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_points, 2)
    pts = circle * (n_std * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T  # (n_points, 2)
    return np.concatenate([pts, pts[:1]], axis=0)  # (n_points + 1, 2) closed polygon
