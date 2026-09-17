"""Pinhole projection still: world point P, camera centre, image plane at z = f, projected point p.

Render:  manim -qm -s --format=png manim/scenes/part04_pinhole.py PinholeProjection
"""
from manim import (BLUE, DOWN, GREEN, LEFT, ORANGE, RIGHT, UP, WHITE, YELLOW, Arrow, Dot, Line, Polygon, Scene, Text,
                   DashedLine, VGroup)


class PinholeProjection(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        centre = LEFT * 5 + DOWN * 1
        # optical axis
        axis = Arrow(centre, RIGHT * 6 + DOWN * 1, buff=0, color="#1f2937", stroke_width=2)
        # image plane at distance f
        plane = Polygon(LEFT * 2.5 + DOWN * 3, LEFT * 2.5 + UP * 1.5, LEFT * 1.5 + UP * 2.2, LEFT * 1.5 + DOWN * 2.3,
                        color=BLUE, fill_opacity=0.15, stroke_width=2)
        # world point and its projection
        P = RIGHT * 4 + UP * 2
        ray = Line(centre, P, color=ORANGE, stroke_width=3)
        p = centre + (P - centre) * (2.5 / 9.0)  # intersection with the plane (f / Z along the ray)
        dot_P = Dot(P, color=ORANGE, radius=0.09)
        dot_p = Dot(p, color=GREEN, radius=0.09)
        dot_c = Dot(centre, color="#1f2937", radius=0.09)
        drop = DashedLine(P, RIGHT * 4 + DOWN * 1, color="#6b7280")
        f_line = DashedLine(centre + DOWN * 1.6, LEFT * 2.5 + DOWN * 2.6, color="#6b7280")
        labels = VGroup(
            Text("camera centre C", font_size=22, color="#1f2937").next_to(dot_c, DOWN),
            Text("image plane (z = f)", font_size=22, color=BLUE).next_to(plane, UP),
            Text("P = (X, Y, Z)", font_size=22, color=ORANGE).next_to(dot_P, UP),
            Text("p = (f X/Z, f Y/Z)", font_size=22, color=GREEN).next_to(dot_p, LEFT),
            Text("optical axis (z)", font_size=20, color="#1f2937").next_to(axis.get_end(), DOWN),
            Text("f", font_size=22, color="#6b7280").next_to(f_line, DOWN),
            Text("Z", font_size=22, color="#6b7280").next_to(drop, RIGHT),
            Text("u = f_x X/Z + c_x,   v = f_y Y/Z + c_y", font_size=24, color="#1f2937").to_edge(DOWN),
        )
        self.add(axis, plane, ray, drop, f_line, dot_c, dot_P, dot_p, labels)
