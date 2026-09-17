"""Manim still: CLIP's contrastive similarity matrix, positives on the diagonal.

Render:  manim -qm -s --format=png manim/scenes/part08_clip_matrix.py ClipContrastiveMatrix
Uses Text only (no LaTeX in the build environment).
"""
from manim import (
    BLUE_E,
    DOWN,
    GREEN_E,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Scene,
    Square,
    Text,
    VGroup,
)


class ClipContrastiveMatrix(Scene):
    def construct(self):
        B = 5
        cell = 0.75
        grid = VGroup()
        for i in range(B):
            for j in range(B):
                sq = Square(side_length=cell, stroke_color=WHITE, stroke_width=1)
                sq.set_fill(GREEN_E if i == j else BLUE_E, opacity=0.85 if i == j else 0.35)
                sq.move_to([j * cell, -i * cell, 0])
                grid.add(sq)
                lbl = Text("+" if i == j else "−", font_size=22).move_to(sq.get_center())
                grid.add(lbl)
        grid.move_to([0.6, 0, 0])
        for i in range(B):
            grid.add(Text(f"v{i}", font_size=20).next_to(grid[2 * (i * B)], LEFT, buff=0.25))
            grid.add(Text(f"t{i}", font_size=20).next_to(grid[2 * i], UP, buff=0.2))
        imgs = Text("image encoder → v_i", font_size=22, color=YELLOW).to_edge(LEFT).shift(UP * 0.2)
        txts = Text("text encoder → t_j", font_size=22, color=YELLOW).to_edge(UP).shift(RIGHT * 1.0 + DOWN * 0.2)
        a1 = Arrow(imgs.get_right(), imgs.get_right() + RIGHT * 0.8, buff=0.05, stroke_width=2)
        a2 = Arrow(txts.get_bottom(), txts.get_bottom() + DOWN * 0.8, buff=0.05, stroke_width=2)
        formula = Text("S_ij = v_i · t_j / τ      L = ½[CE over rows + CE over columns]", font_size=22)
        formula.next_to(grid, DOWN, buff=0.6)
        note = Text("B positives on the diagonal, B² − B negatives everywhere else", font_size=20)
        note.next_to(formula, DOWN, buff=0.25)
        self.add(grid, imgs, txts, a1, a2, formula, note)
