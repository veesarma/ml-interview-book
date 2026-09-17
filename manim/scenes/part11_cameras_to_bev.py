"""Manim still: six cameras on a rig lifting frustum features into a shared BEV grid.

Render with:
    manim -qm -s --format=png manim/scenes/part11_cameras_to_bev.py CamerasToBEV
then copy the PNG to docs/assets/figures/part11_cameras_to_bev.png

Uses Text only (no LaTeX in this environment).
"""
from __future__ import annotations

import numpy as np
from manim import (
    BLUE_D,
    DOWN,
    GREY_B,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    YELLOW_E,
    Circle,
    Dot,
    Line,
    Polygon,
    Scene,
    Square,
    Text,
    VGroup,
    config,
)

CAM_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


class CamerasToBEV(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        config.background_color = WHITE

        title = Text("Six cameras lift into one bird's-eye-view grid", font_size=30, color="#222222")
        title.to_edge(UP, buff=0.35)
        self.add(title)

        # --- the BEV grid -------------------------------------------------------------
        grid = VGroup()
        n, cell = 12, 0.36
        for i in range(n):
            for j in range(n):
                sq = Square(side_length=cell, stroke_width=0.8, stroke_color=GREY_B, fill_opacity=0.0)
                sq.move_to(np.array([(j - (n - 1) / 2) * cell, (i - (n - 1) / 2) * cell, 0.0]))
                grid.add(sq)
        grid.shift(DOWN * 0.12)
        self.add(grid)

        centre = grid.get_center()

        # --- the ego vehicle ----------------------------------------------------------
        ego = Square(side_length=0.3, fill_opacity=1.0, fill_color="#222222", stroke_width=0)
        ego.move_to(centre)
        self.add(ego)

        # --- six camera frusta, 70 degrees each ---------------------------------------
        fov = np.deg2rad(70.0)
        reach = 2.05
        for i in range(6):
            yaw = np.pi / 2 - 2.0 * np.pi * i / 6          # camera 0 points up (ego forward)
            origin = centre + 0.22 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
            left = origin + reach * np.array([np.cos(yaw + fov / 2), np.sin(yaw + fov / 2), 0.0])
            right = origin + reach * np.array([np.cos(yaw - fov / 2), np.sin(yaw - fov / 2), 0.0])
            frustum = Polygon(origin, left, right, stroke_width=2.0,
                              stroke_color=CAM_COLORS[i], fill_color=CAM_COLORS[i], fill_opacity=0.18)
            self.add(frustum)
            cam_dot = Dot(origin, radius=0.055, color=CAM_COLORS[i])
            self.add(cam_dot)
            label = Text(f"cam {i}", font_size=17, color=CAM_COLORS[i])
            label.move_to(origin + 2.38 * np.array([np.cos(yaw), np.sin(yaw), 0.0]))
            self.add(label)

        # --- one ray, drawn with its depth bins ---------------------------------------
        yaw0 = np.pi / 2
        ray_origin = centre + 0.22 * np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
        ray_dir = np.array([np.cos(yaw0 - 0.22), np.sin(yaw0 - 0.22), 0.0])
        ray = Line(ray_origin, ray_origin + reach * ray_dir, stroke_width=3.0, color=YELLOW_E)
        self.add(ray)
        for k, frac in enumerate(np.linspace(0.22, 1.0, 6)):
            p = ray_origin + reach * frac * ray_dir
            weight = np.exp(-0.5 * ((frac - 0.62) / 0.16) ** 2)
            self.add(Circle(radius=0.035 + 0.09 * weight, color=YELLOW_E,
                            fill_opacity=0.35 + 0.6 * weight, stroke_width=1.2).move_to(p))

        # --- the annotations ----------------------------------------------------------
        note_left = VGroup(
            Text("one pixel, one ray", font_size=20, color="#222222"),
            Text("circle size = alpha_d,", font_size=17, color="#555555"),
            Text("the predicted depth", font_size=17, color="#555555"),
            Text("distribution over bins", font_size=17, color="#555555"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        note_left.to_edge(LEFT, buff=0.4).shift(UP * 0.55)
        self.add(note_left)

        note_right = VGroup(
            Text("each cell sums every", font_size=17, color="#555555"),
            Text("lifted feature that", font_size=17, color="#555555"),
            Text("falls inside it", font_size=17, color="#555555"),
            Text("(the splat)", font_size=17, color="#555555"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        note_right.to_edge(RIGHT, buff=0.4).shift(UP * 0.55)
        self.add(note_right)

        overlap = Text("cameras overlap only near the frustum edges:\n"
                       "most BEV cells are covered by exactly one camera",
                       font_size=19, color=BLUE_D)
        overlap.to_edge(DOWN, buff=0.12)
        self.add(overlap)
