# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/geometry/bundle_adjustment.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k bundle_adjustment -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py geometry/bundle_adjustment --force

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
    raise NotImplementedError('TODO: implement so3_exp (see the reference in src/mlbook)')

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
        raise NotImplementedError('TODO: implement n_cameras (see the reference in src/mlbook)')

    @property
    def n_points(self) -> int:
        raise NotImplementedError('TODO: implement n_points (see the reference in src/mlbook)')

@dataclass
class BAResult:
    """Refined state plus the cost trace, which is what you actually inspect when it stalls."""
    Rs: np.ndarray
    ts: np.ndarray
    Xs: np.ndarray
    costs: list[float] = field(default_factory=list)
    n_iters: int = 0
    converged: bool = False

def reprojection_residuals(prob: BAProblem) -> tuple[np.ndarray, np.ndarray]:
    """Residual ``pi(K (R X + t)) - p`` for every observation.

    Returns:
        residuals (M, 2) in pixels, and camera-frame points Y (M, 3) which the
        Jacobians need anyway, so they are computed once here.
    """
    raise NotImplementedError('TODO: implement reprojection_residuals (see the reference in src/mlbook)')

def huber_weights(norms: np.ndarray, delta: float) -> np.ndarray:
    """IRLS weights for the Huber kernel: 1 inside the band, ``delta / e`` outside. (M,) -> (M,).

    Minimising ``sum w_m e_m^2`` with these weights held fixed is one IRLS step on the
    Huber cost, which is why a robust bundle costs no more per iteration than a plain one.
    """
    raise NotImplementedError('TODO: implement huber_weights (see the reference in src/mlbook)')

def huber_cost(norms: np.ndarray, delta: float) -> float:
    """Total Huber cost ``sum rho(e_m)``: quadratic inside the band, linear outside."""
    raise NotImplementedError('TODO: implement huber_cost (see the reference in src/mlbook)')

def observation_jacobians(prob: BAProblem, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Jacobians of every residual w.r.t. its camera and its point.

    With the left perturbation $Y' = \\exp([\\delta\\omega]_\\times) Y + \\delta t$,

    $$\\frac{\\partial Y}{\\partial \\delta\\omega} = -[Y]_\\times, \\qquad
      \\frac{\\partial Y}{\\partial \\delta t} = I_3, \\qquad
      \\frac{\\partial Y}{\\partial X} = R,$$

    and the projection contributes
    $\\partial (u,v) / \\partial Y = \\begin{bmatrix} f_x & s \\\\ 0 & f_y \\end{bmatrix}
     \\frac{1}{Z}\\begin{bmatrix} 1 & 0 & -x \\\\ 0 & 1 & -y \\end{bmatrix}$
    with $x = X_c/Z_c$, $y = Y_c/Z_c$.

    Args:
        Y: (M, 3) camera-frame points from :func:`reprojection_residuals`.
    Returns:
        J_cam (M, 2, 6) ordered ``[delta_omega | delta_t]``, and J_pt (M, 2, 3).
    """
    raise NotImplementedError('TODO: implement observation_jacobians (see the reference in src/mlbook)')

def _normal_equations(prob: BAProblem, r: np.ndarray, w: np.ndarray, J_cam: np.ndarray, J_pt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate the arrowhead Hessian blocks and gradient of ``0.5 sum w ||r||^2``.

    Returns ``B`` (C, 6, 6) camera blocks, ``Cb`` (P, 3, 3) point blocks, ``E`` (C, P, 6, 3)
    coupling blocks, and the gradients ``g_c`` (C, 6), ``g_p`` (P, 3) of the *negative*
    gradient (so the step solves ``H d = g``).
    """
    raise NotImplementedError('TODO: implement _normal_equations (see the reference in src/mlbook)')

def _schur_step(B: np.ndarray, Cb: np.ndarray, E: np.ndarray, g_c: np.ndarray, g_p: np.ndarray, lam: float, free_cams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    raise NotImplementedError('TODO: implement _schur_step (see the reference in src/mlbook)')

def bundle_adjust(prob: BAProblem, max_iters: int=30, huber_delta: float=2.0, fixed_cameras: tuple[int, ...]=(0,), fixed_points: tuple[int, ...]=(), tol: float=1e-10) -> BAResult:
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
    raise NotImplementedError('TODO: implement bundle_adjust (see the reference in src/mlbook)')
