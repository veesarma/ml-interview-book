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
    h, w = image_hw
    f = (w / 2.0) / np.tan(np.deg2rad(fov_x_deg) / 2.0)
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])  # (3, 3)
    return K


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform ``T = [R t; 0 1]``:  ``T^{-1} = [R^T  −R^T t; 0 1]``."""
    R = T[:3, :3]  # (3, 3)
    t = T[:3, 3]  # (3,)
    Tinv = np.eye(4)  # (4, 4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def rotation_z(yaw: float) -> np.ndarray:
    """Rotation about the ego z (up) axis by ``yaw`` radians, (3, 3)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # (3, 3)


def ego_motion_transform(dx: float, dy: float, dyaw: float) -> np.ndarray:
    """``T_curr_from_prev`` for a planar ego motion.

    The ego moved by ``(dx, dy)`` (expressed in the *previous* ego frame) and turned by
    ``dyaw``.  A point fixed in the world with previous-frame coordinates ``p_prev`` has
    current-frame coordinates ``p_curr = R(−dyaw) (p_prev − [dx, dy, 0])``.  Returns (4, 4).
    """
    T = np.eye(4)  # (4, 4)
    R = rotation_z(-dyaw)  # (3, 3)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array([dx, dy, 0.0])
    return T


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
    name: str = "cam"

    @property
    def T_ego_from_cam(self) -> np.ndarray:
        """Camera pose in the ego frame, (4, 4)."""
        return se3_inverse(self.T_cam_from_ego)

    def project(self, points_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project ego-frame points to pixels.

        Args:
            points_ego: (N, 3).
        Returns:
            pixels (N, 2) as (u, v); depth (N,) = z in the camera frame;
            valid (N,) bool = in front of the camera and inside the image.
        """
        R = self.T_cam_from_ego[:3, :3]  # (3, 3)
        t = self.T_cam_from_ego[:3, 3]  # (3,)
        p_cam = points_ego @ R.T + t  # (N, 3)
        depth = p_cam[:, 2]  # (N,)
        z_safe = np.where(depth > 1e-6, depth, 1e-6)  # (N,) avoid divide-by-zero behind camera
        u = self.K[0, 0] * p_cam[:, 0] / z_safe + self.K[0, 2]  # (N,)
        v = self.K[1, 1] * p_cam[:, 1] / z_safe + self.K[1, 2]  # (N,)
        pixels = np.stack([u, v], axis=1)  # (N, 2)
        h, w = self.image_hw
        valid = (depth > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)  # (N,)
        return pixels, depth, valid

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift pixels at known depth back to the ego frame (inverse of ``project``).

        ``x_cam = (u − c_x) d / f_x``, ``y_cam = (v − c_y) d / f_y``, ``z_cam = d``, then
        ``p_ego = R_ego_from_cam p_cam + t_ego_from_cam``.

        Args:
            pixels: (N, 2); depth: (N,).
        Returns:
            (N, 3) ego-frame points.
        """
        x = (pixels[:, 0] - self.K[0, 2]) * depth / self.K[0, 0]  # (N,)
        y = (pixels[:, 1] - self.K[1, 2]) * depth / self.K[1, 1]  # (N,)
        p_cam = np.stack([x, y, depth], axis=1)  # (N, 3)
        T = self.T_ego_from_cam  # (4, 4)
        return p_cam @ T[:3, :3].T + T[:3, 3]  # (N, 3)


def camera_pose_from_yaw(yaw: float, position: np.ndarray) -> np.ndarray:
    """``T_ego_from_cam`` for a camera at ``position`` (3,) looking horizontally at ``yaw``.

    The camera's optical axis (cam z) is the ego direction ``(cos yaw, sin yaw, 0)``,
    cam x (image right) is ``(sin yaw, −cos yaw, 0)`` and cam y (image down) is ``(0, 0, −1)``.
    The columns of ``R_ego_from_cam`` are the camera axes expressed in the ego frame.
    """
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # (3,) cam z in ego
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # (3,) cam x in ego
    down = np.array([0.0, 0.0, -1.0])  # (3,) cam y in ego
    R = np.stack([right, down, forward], axis=1)  # (3, 3) columns = cam axes
    T = np.eye(4)  # (4, 4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def make_surround_rig(
    n_cams: int = 6, fov_x_deg: float = 70.0, image_hw: tuple[int, int] = (256, 448),
    radius: float = 1.0, height: float = 1.6,
) -> list[Camera]:
    """``n_cams`` cameras spread evenly in yaw on a circle of ``radius`` around the ego origin.

    Camera 0 faces forward (+x); yaws increase counter-clockwise (towards +y = left).
    """
    K = intrinsics_from_fov(image_hw, fov_x_deg)  # (3, 3)
    cams: list[Camera] = []
    for i in range(n_cams):
        yaw = 2.0 * np.pi * i / n_cams
        position = np.array([radius * np.cos(yaw), radius * np.sin(yaw), height])  # (3,)
        T_ego_from_cam = camera_pose_from_yaw(yaw, position)  # (4, 4)
        cams.append(Camera(K=K, T_cam_from_ego=se3_inverse(T_ego_from_cam), image_hw=image_hw, name=f"cam{i}"))
    return cams


def rolling_shutter_row_time(v: np.ndarray, image_h: int, readout_s: float) -> np.ndarray:
    """Capture time offset of image row ``v`` for a rolling shutter that reads top→bottom.

    ``t(v) = readout_s · v / (H − 1)``.  A point moving at ego-relative speed ``s`` is
    displaced by ``s · t(v)`` between the first and the observed row — the origin of the
    "leaning pole" artefact and of why row timestamps must enter the projection.
    """
    return readout_s * v / max(image_h - 1, 1)  # (N,)
