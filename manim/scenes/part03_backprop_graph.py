"""Manim still: computational graph of Z = XW + b -> H = relu(Z) -> L = CE(H W2, y)
with forward values on top and gradient flow (red) underneath.

Render:  manim -qm -s --format=png manim/scenes/part03_backprop_graph.py BackpropGraph
Uses Text only (no LaTeX in the build environment).
"""
from manim import (BLACK, BLUE_E, DOWN, GREY_D, LEFT, ORANGE, RED_E, RIGHT, UP, WHITE, Arrow, Create, RoundedRectangle,
                   Scene, Text, VGroup, config)

config.background_color = WHITE


def node(label: str, shape: str, color) -> VGroup:
    box = RoundedRectangle(width=2.45, height=1.0, corner_radius=0.15, color=color, fill_color=color, fill_opacity=0.12)
    t1 = Text(label, font_size=17, color=BLACK, font="monospace").move_to(box.get_center() + 0.15 * UP)
    t2 = Text(shape, font_size=16, color=GREY_D).move_to(box.get_center() + 0.28 * DOWN)
    return VGroup(box, t1, t2)


class BackpropGraph(Scene):
    def construct(self) -> None:
        title = Text("Reverse-mode AD: forward stores, backward multiplies by local Jacobians",
                     font_size=24, color=BLACK).to_edge(UP)
        specs = [("X", "(N, d)"), ("Z = XW + b", "(N, h)"), ("H = relu(Z)", "(N, h)"),
                 ("S = H W2", "(N, K)"), ("L = CE(S, y)", "()")]
        nodes = VGroup(*[node(a, b, BLUE_E) for a, b in specs]).arrange(RIGHT, buff=0.3).shift(0.8 * UP)
        params = VGroup(node("W (d, h)", "b (h,)", ORANGE), node("W2 (h, K)", "", ORANGE))
        params[0].next_to(nodes[1], UP, buff=0.5)
        params[1].next_to(nodes[3], UP, buff=0.5)
        fwd_arrows = VGroup(*[Arrow(nodes[i].get_right(), nodes[i + 1].get_left(), buff=0.02, color=BLUE_E,
                                    stroke_width=3, max_tip_length_to_length_ratio=0.6) for i in range(4)])
        p_arrows = VGroup(Arrow(params[0].get_bottom(), nodes[1].get_top(), buff=0.05, color=ORANGE, stroke_width=3),
                          Arrow(params[1].get_bottom(), nodes[3].get_top(), buff=0.05, color=ORANGE, stroke_width=3))
        grads = [("dX = dZ W^T", "(N, d)"), ("dZ = dH * 1[Z>0]", "(N, h)"), ("dH = dS W2^T", "(N, h)"),
                 ("dS = (P - Y)/N", "(N, K)"), ("dL = 1", "()")]
        gnodes = VGroup(*[node(a, b, RED_E) for a, b in grads]).arrange(RIGHT, buff=0.3).shift(1.6 * DOWN)
        bwd_arrows = VGroup(*[Arrow(gnodes[i + 1].get_left(), gnodes[i].get_right(), buff=0.02, color=RED_E,
                                    stroke_width=3, max_tip_length_to_length_ratio=0.6) for i in range(4)])
        pgrad = Text("parameter grads:  dW = X^T dZ     db = sum_n dZ     dW2 = H^T dS", font_size=18, color=RED_E, font="monospace")
        pgrad.next_to(gnodes, DOWN, buff=0.4)
        self.add(title, nodes, params, fwd_arrows, p_arrows, gnodes, bwd_arrows, pgrad)
        self.play(Create(bwd_arrows))
