"""Megatron tensor-parallel MLP: column-sharded A, row-sharded B, one all-reduce.

Render a still:  manim -qm -s --format=png manim/scenes/part14_tensor_parallel.py TensorParallelMLP
"""

from manim import (BLUE, DOWN, GREEN, LEFT, ORANGE, RIGHT, UP, WHITE, YELLOW, Arrow, Rectangle, Scene, Text, VGroup)


class TensorParallelMLP(Scene):
    def construct(self):
        title = Text("Tensor-parallel MLP:  Y = GELU(X A) B", font_size=32).to_edge(UP)
        self.add(title)
        x = Rectangle(width=1.2, height=1.6, color=WHITE).shift(LEFT * 5.5)
        x_label = Text("X\n(N, h)\nreplicated", font_size=18).next_to(x, DOWN)
        self.add(x, x_label)
        rows = VGroup()
        for i, color in enumerate([BLUE, GREEN]):
            y = UP * 1.1 - i * UP * 2.2
            a = Rectangle(width=1.4, height=1.2, color=color).shift(LEFT * 2.5 + y)
            a_l = Text(f"A_{i}  (h, d_ff/2)\ncolumn shard", font_size=16).next_to(a, DOWN, buff=0.1)
            g = Text("GELU", font_size=18).shift(LEFT * 0.2 + y)
            b = Rectangle(width=1.4, height=1.2, color=color).shift(RIGHT * 1.8 + y)
            b_l = Text(f"B_{i}  (d_ff/2, h)\nrow shard", font_size=16).next_to(b, DOWN, buff=0.1)
            gpu = Text(f"GPU {i}", font_size=18, color=color).next_to(a, LEFT, buff=0.2).shift(UP * 0.8)
            rows.add(a, a_l, g, b, b_l, gpu)
            self.add(Arrow(x.get_right(), a.get_left(), buff=0.1, color=color))
            self.add(Arrow(a.get_right(), g.get_left(), buff=0.1, color=color))
            self.add(Arrow(g.get_right(), b.get_left(), buff=0.1, color=color))
        self.add(rows)
        ar = Rectangle(width=1.9, height=1.4, color=YELLOW).shift(RIGHT * 4.6)
        ar_l = Text("all-reduce\n(sum partials)", font_size=18, color=YELLOW).move_to(ar)
        out = Text("Y (N, h)\non every GPU", font_size=18).next_to(ar, DOWN, buff=0.15)
        for i in range(2):
            self.add(Arrow(rows[3 + 6 * i].get_right(), ar.get_left(), buff=0.1, color=ORANGE))
        note = Text("no communication between A_i and B_i (GELU is element-wise);  1 all-reduce forward, 1 backward", font_size=18).to_edge(DOWN)
        self.add(ar, ar_l, out, note)
