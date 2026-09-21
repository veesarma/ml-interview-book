"""Bundle adjustment: jointly refine camera poses and 3-D points (NumPy).

Bundle adjustment is the least-squares problem at the end of every structure-from-motion
pipeline.  Given observations $p_{ij}$ of point $j$ in camera $i$, minimise total
reprojection error over poses and structure at once:

$$\\min_{\\{R_i, t_i\\},\\, \\{X_j\\}} \\sum_{(i,j) \\in \\mathcal{O}} \\rho\\big(\\| \\pi(K_i (R_i X_j + t_i)) - p_{ij} \\|\\big).$$

Three implementation facts carry the whole method:

* **Local parametrisation.** Rotations live on SO(3), so the update is a 3-vector
  $\\delta\\omega$ applied as $R \\leftarrow \\exp([\\delta\\omega]_\\times) R$.  No quaternion
  renormalisation, no gimbal lock, and the Jacobian stays exact to first order.
* **Sparsity.** A camera block and a point block interact only where that camera saw
  that point, so the Hessian is arrowhead-shaped.  The Schur complement eliminates the
  points analytically and leaves a system the size of the cameras alone.
* **Robustness.** One surviving mismatch costs $O(e^2)$ under plain least squares and
  drags every pose.  A Huber kernel caps its gradient.

Conventions match ``mlbook.geometry.camera``: ``P_c = R P_w + t``, pixels are ``(N, 2)``
rows, and the camera looks down +z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import skew_symmetric


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential ``exp([w]_x)``: a rotation vector (3,) to a matrix (3, 3).

    ``|w|`` is the angle, ``w / |w|`` the axis.  The series is expanded near zero so the
    tiny increments that bundle adjustment actually takes stay numerically clean.
    """
    theta = float(np.linalg.norm(w))
    W = skew_symmetric(w)  # (3, 3)
    if theta < 1e-8:
        return np.eye(3) + W + 0.5 * (W @ W)  # (3, 3) second-order series
    return np.eye(3) + (np.sin(theta) / theta) * W + ((1.0 - np.cos(theta)) / theta**2) * (W @ W)


@dataclass
class BAProblem:
    """A bundle adjustment problem: cameras, points, and who saw whom.

    Attributes:
        Rs: (C, 3, 3) rotations, ``P_c = R P_w + t``.
        ts: (C, 3) translations.
        Ks: (C, 3, 3) intrinsics, one per camera (pass the same matrix C times if shared).
        Xs: (P, 3) world points.
        cam_idx: (M,) int, which camera made observation m.
        pt_idx: (M,) int, which point observation m is of.
        pixels: (M, 2) measured pixel coordinates.
    """

    Rs: np.ndarray
    ts: np.ndarray
    Ks: np.ndarray
    Xs: np.ndarray
    cam_idx: np.ndarray
    pt_idx: np.ndarray
    pixels: np.ndarray

    @property
    def n_cameras(self) -> int:
        return len(self.Rs)

    @property
    def n_points(self) -> int:
        return len(self.Xs)


@dataclass
class BAResult:
    """Refined state plus the cost trace, which is what you actually inspect when it stalls."""

    Rs: np.ndarray            # (C, 3, 3)
    ts: np.ndarray            # (C, 3)
    Xs: np.ndarray            # (P, 3)
    costs: list[float] = field(default_factory=list)   # robust cost after each accepted step
    n_iters: int = 0
    converged: bool = False


def reprojection_residuals(prob: BAProblem) -> tuple[np.ndarray, np.ndarray]:
    """Residual ``pi(K (R X + t)) - p`` for every observation.

    Returns:
        residuals (M, 2) in pixels, and camera-frame points Y (M, 3) which the
        Jacobians need anyway, so they are computed once here.
    """
    R_m = prob.Rs[prob.cam_idx]        # (M, 3, 3) rotation of the observing camera
    t_m = prob.ts[prob.cam_idx]        # (M, 3)
    K_m = prob.Ks[prob.cam_idx]        # (M, 3, 3)
    X_m = prob.Xs[prob.pt_idx]         # (M, 3) world point being observed
    Y = np.einsum("mij,mj->mi", R_m, X_m) + t_m  # (M, 3) camera-frame point: R X + t
    uv_h = np.einsum("mij,mj->mi", K_m, Y)       # (M, 3) homogeneous pixel K Y
    uv = uv_h[:, :2] / uv_h[:, 2:3]              # (M, 2) perspective divide
    return uv - prob.pixels, Y                   # (M, 2), (M, 3)


def huber_weights(norms: np.ndarray, delta: float) -> np.ndarray:
    """IRLS weights for the Huber kernel: 1 inside the band, ``delta / e`` outside. (M,) -> (M,).

    Minimising ``sum w_m e_m^2`` with these weights held fixed is one IRLS step on the
    Huber cost, which is why a robust bundle costs no more per iteration than a plain one.
    """
    return np.where(norms <= delta, 1.0, delta / np.maximum(norms, 1e-12))  # (M,)


def huber_cost(norms: np.ndarray, delta: float) -> float:
    """Total Huber cost ``sum rho(e_m)``: quadratic inside the band, linear outside."""
    quad = 0.5 * norms**2
    lin = delta * (norms - 0.5 * delta)
    return float(np.where(norms <= delta, quad, lin).sum())


def observation_jacobians(prob: BAProblem, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Analytic Jacobians of every residual w.r.t. its camera and its point.

    With the left perturbation $Y' = \exp([\delta\omega]_\times) Y + \delta t$,

    $$\frac{\partial Y}{\partial \delta\omega} = -[Y]_\times, \qquad
      \frac{\partial Y}{\partial \delta t} = I_3, \qquad
      \frac{\partial Y}{\partial X} = R,$$

    and the projection contributes
    $\partial (u,v) / \partial Y = \begin{bmatrix} f_x & s \\ 0 & f_y \end{bmatrix}
     \frac{1}{Z}\begin{bmatrix} 1 & 0 & -x \\ 0 & 1 & -y \end{bmatrix}$
    with $x = X_c/Z_c$, $y = Y_c/Z_c$.

    Args:
        Y: (M, 3) camera-frame points from :func:`reprojection_residuals`.
    Returns:
        J_cam (M, 2, 6) ordered ``[delta_omega | delta_t]``, and J_pt (M, 2, 3).
    """
    K_m = prob.Ks[prob.cam_idx]              # (M, 3, 3)
    Z = Y[:, 2:3]                            # (M, 1) depth
    xy = Y[:, :2] / Z                        # (M, 2) normalised image coordinates
    M = len(Y)
    d_norm = np.zeros((M, 2, 3))             # (M, 2, 3) d(x, y) / dY
    d_norm[:, 0, 0] = d_norm[:, 1, 1] = 1.0 / Z[:, 0]
    d_norm[:, 0, 2] = -xy[:, 0] / Z[:, 0]
    d_norm[:, 1, 2] = -xy[:, 1] / Z[:, 0]
    K2 = K_m[:, :2, :2]                      # (M, 2, 2) [[fx, s], [0, fy]]
    J_proj = K2 @ d_norm                     # (M, 2, 3) d(u, v) / dY

    dY_domega = -np.stack([skew_symmetric(y) for y in Y])  # (M, 3, 3) = -[Y]_x
    J_omega = J_proj @ dY_domega             # (M, 2, 3)
    J_cam = np.concatenate([J_omega, J_proj], axis=2)      # (M, 2, 6) [omega | t]
    J_pt = J_proj @ prob.Rs[prob.cam_idx]    # (M, 2, 3) chain through dY/dX = R
    return J_cam, J_pt


def _normal_equations(
    prob: BAProblem, r: np.ndarray, w: np.ndarray, J_cam: np.ndarray, J_pt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate the arrowhead Hessian blocks and gradient of ``0.5 sum w ||r||^2``.

    Returns ``B`` (C, 6, 6) camera blocks, ``Cb`` (P, 3, 3) point blocks, ``E`` (C, P, 6, 3)
    coupling blocks, and the gradients ``g_c`` (C, 6), ``g_p`` (P, 3) of the *negative*
    gradient (so the step solves ``H d = g``).
    """
    C, P = prob.n_cameras, prob.n_points
    wc = w[:, None, None]                                   # (M, 1, 1) broadcast weight
    B = np.zeros((C, 6, 6))
    Cb = np.zeros((P, 3, 3))
    E = np.zeros((C, P, 6, 3))
    g_c = np.zeros((C, 6))
    g_p = np.zeros((P, 3))
    np.add.at(B, prob.cam_idx, wc * (J_cam.transpose(0, 2, 1) @ J_cam))      # (M,6,6) scattered
    np.add.at(Cb, prob.pt_idx, wc * (J_pt.transpose(0, 2, 1) @ J_pt))        # (M,3,3)
    np.add.at(E, (prob.cam_idx, prob.pt_idx), wc * (J_cam.transpose(0, 2, 1) @ J_pt))  # (M,6,3)
    np.add.at(g_c, prob.cam_idx, -(w[:, None] * np.einsum("mab,ma->mb", J_cam, r)))
    np.add.at(g_p, prob.pt_idx, -(w[:, None] * np.einsum("mab,ma->mb", J_pt, r)))
    return B, Cb, E, g_c, g_p


def _schur_step(
    B: np.ndarray, Cb: np.ndarray, E: np.ndarray,
    g_c: np.ndarray, g_p: np.ndarray, lam: float, free_cams: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One damped Gauss-Newton step, eliminating the points by Schur complement.

    The normal equations are ``[[B, E], [E^T, C]] [dc; dp] = [g_c; g_p]``.  Because ``C``
    is block diagonal it inverts in ``O(P)``, giving the reduced camera system

    ``(B - E C^-1 E^T) dc = g_c - E C^-1 g_p``,  then  ``dp = C^-1 (g_p - E^T dc)``.

    Damping ``lam`` is added to both diagonals (Levenberg-Marquardt), which also
    regularises the 7-dof gauge freedom that a free-floating reconstruction always has.

    Args:
        free_cams: (F,) indices of cameras that are allowed to move (gauge fixing).
    Returns:
        dc (F, 6) camera increments for ``free_cams``, dp (P, 3) point increments.
    """
    F, P = len(free_cams), len(Cb)
    Bd = B[free_cams] + lam * np.eye(6)                 # (F, 6, 6) damped camera blocks
    Cd = Cb + lam * np.eye(3)                           # (P, 3, 3) damped point blocks
    C_inv = np.linalg.inv(Cd)                           # (P, 3, 3) block-diagonal inverse
    Ef = E[free_cams].transpose(0, 2, 1, 3)             # (F, 6, P, 3) coupling, camera-major

    # Y[f,a,p,:] = sum_b E[f,a,p,b] C_inv[p,b,:]  -> the E C^-1 product, kept blockwise
    Y = np.einsum("fapb,pbc->fapc", Ef, C_inv)          # (F, 6, P, 3)
    # S[f,a,g,b] = -sum_{p,c} Y[f,a,p,c] E[g,b,p,c], then B_d goes on the block diagonal
    S = -np.einsum("fapc,gbpc->fagb", Y, Ef)            # (F, 6, F, 6)
    S[np.arange(F), :, np.arange(F), :] += Bd           # (F, 6, 6) camera blocks
    S = S.reshape(6 * F, 6 * F)                         # (6F, 6F) reduced camera system
    b = g_c[free_cams] - np.einsum("fapc,pc->fa", Y, g_p)   # (F, 6) reduced gradient
    dc = np.linalg.solve(S, b.reshape(-1)).reshape(F, 6)    # (F, 6)
    # back-substitute: dp = C^-1 (g_p - E^T dc)
    Et_dc = np.einsum("fapb,fa->pb", Ef, dc)                # (P, 3)
    dp = np.einsum("pbc,pc->pb", C_inv, g_p - Et_dc)        # (P, 3)
    return dc, dp


def bundle_adjust(
    prob: BAProblem,
    max_iters: int = 30,
    huber_delta: float = 2.0,
    fixed_cameras: tuple[int, ...] = (0,),
    fixed_points: tuple[int, ...] = (),
    tol: float = 1e-10,
) -> BAResult:
    """Levenberg-Marquardt bundle adjustment with a Huber kernel and Schur elimination.

    Args:
        prob: the problem; ``prob`` itself is not mutated.
        huber_delta: robust band in pixels.  Set it near your expected noise (1-3 px).
        fixed_cameras: cameras held still to fix the gauge.  A reconstruction from images
            alone is determined only up to a similarity (7 dof), so freezing one camera
            removes 6 and LM damping absorbs the remaining scale.
        fixed_points: points held still, e.g. surveyed ground control points, which is
            how an aerial reconstruction becomes metric and georeferenced.

    Returns:
        ``BAResult`` with refined state and the robust cost after each accepted step.
    """
    Rs, ts, Xs = prob.Rs.copy(), prob.ts.copy(), prob.Xs.copy()
    free_cams = np.array([i for i in range(prob.n_cameras) if i not in fixed_cameras])
    frozen_pts = np.array(sorted(fixed_points), dtype=int)
    state = BAProblem(Rs, ts, prob.Ks, Xs, prob.cam_idx, prob.pt_idx, prob.pixels)

    r, Y = reprojection_residuals(state)                         # (M, 2), (M, 3)
    cost = huber_cost(np.linalg.norm(r, axis=1), huber_delta)
    result = BAResult(Rs, ts, Xs, [cost])
    lam = 1e-3
    for it in range(max_iters):
        result.n_iters = it + 1
        norms = np.linalg.norm(r, axis=1)                        # (M,)
        w = huber_weights(norms, huber_delta)                    # (M,)
        J_cam, J_pt = observation_jacobians(state, Y)            # (M,2,6), (M,2,3)
        B, Cb, E, g_c, g_p = _normal_equations(state, r, w, J_cam, J_pt)
        accepted = False
        for _ in range(8):                                       # LM inner loop on lambda
            try:
                dc, dp = _schur_step(B, Cb, E, g_c, g_p, lam, free_cams)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            if len(frozen_pts):
                dp[frozen_pts] = 0.0
            Rs_try, ts_try, Xs_try = Rs.copy(), ts.copy(), Xs + dp
            for k, cam in enumerate(free_cams):                  # apply the SE(3) increment
                dR = so3_exp(dc[k, :3])                          # (3, 3)
                Rs_try[cam] = dR @ Rs[cam]
                ts_try[cam] = dR @ ts[cam] + dc[k, 3:]
            trial = BAProblem(Rs_try, ts_try, prob.Ks, Xs_try, prob.cam_idx, prob.pt_idx, prob.pixels)
            r_try, Y_try = reprojection_residuals(trial)
            cost_try = huber_cost(np.linalg.norm(r_try, axis=1), huber_delta)
            if np.isfinite(cost_try) and cost_try < cost:
                Rs, ts, Xs, state, r, Y = Rs_try, ts_try, Xs_try, trial, r_try, Y_try
                lam = max(lam * 0.3, 1e-10)
                accepted, improvement, cost = True, cost - cost_try, cost_try
                break
            lam *= 10.0                                          # step too bold; trust GN less
        if not accepted:
            break
        result.costs.append(cost)
        if improvement < tol * max(cost, 1.0):
            result.converged = True
            break
    result.Rs, result.ts, result.Xs = Rs, ts, Xs
    return result
