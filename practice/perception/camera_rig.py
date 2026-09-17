# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/camera_rig.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k camera_rig -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/camera_rig --force

"""Pinhole cameras, extrinsics and a synthetic surround-view rig (NumPy).

Frames and conventions used throughout Part XI:

* **ego frame**: x forward, y left, z up (right-handed, metres; the nuScenes convention).
* **camera frame** (OpenCV): x right, y down, z forward (optical axis).
* An extrinsic ``T_cam_from_ego`` is a 4×4 homogeneous transform mapping ego-frame
  points to camera-frame points: ``p_cam = R p_ego + t``.
* Intrinsics ``K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]``; projection is
  ``u = f_x · x/z + c_x``, ``v = f_y · y/z + c_y`` (pixels, origin top-left).

Points are stored **row-major**, one point per row: ``P ∈ R^{N×3}``, so we write
``P @ R.T + t`` rather than ``R p + t``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def intrinsics_from_fov(image_hw: tuple[int, int], fov_x_deg: float) -> np.ndarray:
    """Intrinsics of a square-pixel pinhole camera with horizontal field of view ``fov_x``.

    ``f_x = (W / 2) / tan(fov_x / 2)``, ``f_y = f_x`` (square pixels), principal point at the
    image centre.  Returns ``K`` with shape (3, 3).
    """
    raise NotImplementedError('TODO: implement intrinsics_from_fov (see the reference in src/mlbook)')

def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform ``T = [R t; 0 1]``:  ``T^{-1} = [R^T  −R^T t; 0 1]``."""
    raise NotImplementedError('TODO: implement se3_inverse (see the reference in src/mlbook)')

def rotation_z(yaw: float) -> np.ndarray:
    """Rotation about the ego z (up) axis by ``yaw`` radians, (3, 3)."""
    raise NotImplementedError('TODO: implement rotation_z (see the reference in src/mlbook)')

def ego_motion_transform(dx: float, dy: float, dyaw: float) -> np.ndarray:
    """``T_curr_from_prev`` for a planar ego motion.

    The ego moved by ``(dx, dy)`` (expressed in the *previous* ego frame) and turned by
    ``dyaw``.  A point fixed in the world with previous-frame coordinates ``p_prev`` has
    current-frame coordinates ``p_curr = R(−dyaw) (p_prev − [dx, dy, 0])``.  Returns (4, 4).
    """
    raise NotImplementedError('TODO: implement ego_motion_transform (see the reference in src/mlbook)')

@dataclass
class Camera:
    """One calibrated pinhole camera.

    Attributes:
        K: intrinsics (3, 3).
        T_cam_from_ego: extrinsics (4, 4) mapping ego-frame points into the camera frame.
        image_hw: (H, W) in pixels.
        name: label used in figures.
    """
    K: np.ndarray
    T_cam_from_ego: np.ndarray
    image_hw: tuple[int, int]
    name: str = 'cam'

    @property
    def T_ego_from_cam(self) -> np.ndarray:
        """Camera pose in the ego frame, (4, 4)."""
        raise NotImplementedError('TODO: implement T_ego_from_cam (see the reference in src/mlbook)')

    def project(self, points_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project ego-frame points to pixels.

        Args:
            points_ego: (N, 3).
        Returns:
            pixels (N, 2) as (u, v); depth (N,) = z in the camera frame;
            valid (N,) bool = in front of the camera and inside the image.
        """
        raise NotImplementedError('TODO: implement project (see the reference in src/mlbook)')

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift pixels at known depth back to the ego frame (inverse of ``project``).

        ``x_cam = (u − c_x) d / f_x``, ``y_cam = (v − c_y) d / f_y``, ``z_cam = d``, then
        ``p_ego = R_ego_from_cam p_cam + t_ego_from_cam``.

        Args:
            pixels: (N, 2); depth: (N,).
        Returns:
            (N, 3) ego-frame points.
        """
        raise NotImplementedError('TODO: implement unproject (see the reference in src/mlbook)')

def camera_pose_from_yaw(yaw: float, position: np.ndarray) -> np.ndarray:
    """``T_ego_from_cam`` for a camera at ``position`` (3,) looking horizontally at ``yaw``.

    The camera's optical axis (cam z) is the ego direction ``(cos yaw, sin yaw, 0)``,
    cam x (image right) is ``(sin yaw, −cos yaw, 0)`` and cam y (image down) is ``(0, 0, −1)``.
    The columns of ``R_ego_from_cam`` are the camera axes expressed in the ego frame.
    """
    raise NotImplementedError('TODO: implement camera_pose_from_yaw (see the reference in src/mlbook)')

def make_surround_rig(n_cams: int=6, fov_x_deg: float=70.0, image_hw: tuple[int, int]=(256, 448), radius: float=1.0, height: float=1.6) -> list[Camera]:
    """``n_cams`` cameras spread evenly in yaw on a circle of ``radius`` around the ego origin.

    Camera 0 faces forward (+x); yaws increase counter-clockwise (towards +y = left).
    """
    raise NotImplementedError('TODO: implement make_surround_rig (see the reference in src/mlbook)')

def rolling_shutter_row_time(v: np.ndarray, image_h: int, readout_s: float) -> np.ndarray:
    """Capture time offset of image row ``v`` for a rolling shutter that reads top→bottom.

    ``t(v) = readout_s · v / (H − 1)``.  A point moving at ego-relative speed ``s`` is
    displaced by ``s · t(v)`` between the first and the observed row — the origin of the
    "leaning pole" artefact and of why row timestamps must enter the projection.
    """
    raise NotImplementedError('TODO: implement rolling_shutter_row_time (see the reference in src/mlbook)')
