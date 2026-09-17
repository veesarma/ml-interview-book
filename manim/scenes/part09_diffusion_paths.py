"""Manim still: the forward diffusion corridor against the straight rectified-flow path.

Render:  manim -qm -s --format=png manim/scenes/part09_diffusion_paths.py DiffusionVsFlowPaths
Uses Text only (no LaTeX in the build environment).
"""
import numpy as np
from manim import (
    BLUE_E,
    DOWN,
    GREY_B,
    LEFT,
    ORANGE,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Dot,
    Line,
    Scene,
    Text,
    VGroup,
    VMobject,
)


class DiffusionVsFlowPaths(Scene):
    def construct(self):
        rng = np.random.default_rng(0)
        x_left, x_right = -5.2, 5.2

        # the two endpoints: data on the left, noise on the right
        data_pts = VGroup(*[
            Dot([x_left + 0.35 * rng.standard_normal(), 1.6 + 0.35 * rng.standard_normal(), 0],
                radius=0.045, color=YELLOW) for _ in range(28)
        ])
        noise_pts = VGroup(*[
            Dot([x_right + 0.75 * rng.standard_normal(), 1.6 + 0.75 * rng.standard_normal(), 0],
                radius=0.045, color=GREY_B) for _ in range(28)
        ])

        # top: the diffusion trajectory, a random walk that fans out as alpha_bar decays
        diff = VMobject(stroke_color=BLUE_E, stroke_width=3)
        ts = np.linspace(0, 1, 60)
        pts = []
        for t in ts:
            x = x_left + (x_right - x_left) * t
            wiggle = 0.9 * t * np.sin(9 * t) + 0.25 * t * rng.standard_normal()
            pts.append([x, 1.6 + wiggle, 0])
        diff.set_points_smoothly(pts)

        # bottom: the rectified-flow trajectory, a straight line between the same endpoints
        flow = Line([x_left, -1.6, 0], [x_right, -1.6, 0], stroke_color=ORANGE, stroke_width=3)
        data_pts2 = VGroup(*[
            Dot([x_left + 0.35 * rng.standard_normal(), -1.6 + 0.35 * rng.standard_normal(), 0],
                radius=0.045, color=YELLOW) for _ in range(28)
        ])
        noise_pts2 = VGroup(*[
            Dot([x_right + 0.75 * rng.standard_normal(), -1.6 + 0.75 * rng.standard_normal(), 0],
                radius=0.045, color=GREY_B) for _ in range(28)
        ])

        # Euler steps drawn on both: 4 steps, same budget, different error
        steps_d = VGroup()
        for i in range(4):
            t0, t1 = i / 4, (i + 1) / 4
            p0 = np.array(pts[int(t0 * 59)])
            p1 = np.array(pts[int(t1 * 59)])
            steps_d.add(Arrow(p0, p1, buff=0, stroke_width=2.5, color=WHITE,
                              max_tip_length_to_length_ratio=0.12))
        steps_f = VGroup()
        for i in range(4):
            p0 = np.array([x_left + (x_right - x_left) * i / 4, -1.6, 0])
            p1 = np.array([x_left + (x_right - x_left) * (i + 1) / 4, -1.6, 0])
            steps_f.add(Arrow(p0, p1, buff=0, stroke_width=2.5, color=WHITE,
                              max_tip_length_to_length_ratio=0.12))

        title = Text("Same two distributions, two paths between them", font_size=30).to_edge(UP, buff=0.35)
        lab_diff = Text("diffusion: the marginal path curves,", font_size=22, color=BLUE_E)
        lab_diff2 = Text("so 4 Euler steps leave the corridor", font_size=22, color=BLUE_E)
        lab_diff.move_to([-0.4, 0.75, 0])
        lab_diff2.next_to(lab_diff, DOWN, buff=0.12)
        lab_flow = Text("rectified flow: x_t = (1 - t) x_0 + t x_1,", font_size=22, color=ORANGE)
        lab_flow2 = Text("straight, so few steps are enough", font_size=22, color=ORANGE)
        lab_flow.next_to(flow, DOWN, buff=0.55).shift(LEFT * 0.3)
        lab_flow2.next_to(lab_flow, DOWN, buff=0.12)

        left_tag = Text("data", font_size=22, color=YELLOW).move_to([x_left, 0.0, 0])
        right_tag = Text("noise", font_size=22, color=GREY_B).move_to([x_right, 0.0, 0])

        self.add(title, diff, flow, steps_d, steps_f,
                 data_pts, noise_pts, data_pts2, noise_pts2,
                 lab_diff, lab_diff2, lab_flow, lab_flow2, left_tag, right_tag)
