"""SVD as rotate -> scale -> rotate, shown on the unit circle.

Render a still:  manim -qm -s --format=png manim/scenes/part01_svd.py SVDRotateScaleRotate
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY,
    LEFT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Circle,
    Scene,
    Text,
    VGroup,
    NumberPlane,
    Polygon,
)


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x2 matrix to an (n, 3) array of manim points (z untouched)."""
    out = pts.copy()
    out[:, :2] = pts[:, :2] @ M.T
    return out


class SVDRotateScaleRotate(Scene):
    def construct(self) -> None:
        A = np.array([[2.0, 1.0], [0.5, 1.5]])  # (2, 2)
        U, s, Vt = np.linalg.svd(A)  # (2,2), (2,), (2,2)
        S = np.diag(s)
        stages = [
            ("unit circle", np.eye(2)),
            ("Vᵀ: rotate", Vt),
            ("S Vᵀ: scale by σ₁, σ₂", S @ Vt),
            ("U S Vᵀ = A: rotate", U @ S @ Vt),
        ]
        theta = np.linspace(0, 2 * np.pi, 120)
        circle_pts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)  # (120, 3)
        panels = VGroup()
        for i, (label, M) in enumerate(stages):
            pts = _apply(M, circle_pts) * 0.9
            shape = Polygon(*pts, color=BLUE, fill_opacity=0.25, stroke_width=3)
            e1 = Arrow(start=np.zeros(3), end=_apply(M, np.array([[1.0, 0.0, 0.0]]))[0] * 0.9, buff=0, color=RED, stroke_width=4)
            e2 = Arrow(start=np.zeros(3), end=_apply(M, np.array([[0.0, 1.0, 0.0]]))[0] * 0.9, buff=0, color=YELLOW, stroke_width=4)
            axes = NumberPlane(x_range=[-2.5, 2.5, 1], y_range=[-2.5, 2.5, 1], x_length=3.0, y_length=3.0,
                               background_line_style={"stroke_color": GREY, "stroke_width": 1, "stroke_opacity": 0.3})
            title = Text(label, font_size=22).next_to(axes, UP, buff=0.15)
            panel = VGroup(axes, shape, e1, e2, title)
            panels.add(panel)
        panels.arrange(RIGHT, buff=0.3).scale_to_fit_width(13.0).move_to(np.zeros(3))
        header = Text("A = U S Vᵀ : every matrix is rotate → scale → rotate", font_size=30).to_edge(UP)
        footer = Text(f"σ₁ = {s[0]:.2f}, σ₂ = {s[1]:.2f};  spectral norm = σ₁, Frobenius² = σ₁² + σ₂²", font_size=22).to_edge(DOWN)
        self.add(header, panels, footer)
