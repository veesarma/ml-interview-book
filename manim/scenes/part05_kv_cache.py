"""Causal attention with a KV cache: the score matrix during prefill (4 prompt tokens) and two
decode steps. Cached keys are the columns; the new query row attends to all of them; the masked
upper triangle never exists at decode time because there is only one query.

Render a still:  manim -qm -s --format=png manim/scenes/part05_kv_cache.py KVCacheDecode
"""
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Scene,
    Square,
    Text,
    VGroup,
)

CELL = 0.42


def grid(T_q: int, T_k: int, q_offset: int, cached: int) -> VGroup:
    """T_q query rows vs T_k key columns; query i (absolute q_offset + i) sees key j iff j <= q_offset + i."""
    g = VGroup()
    for i in range(T_q):
        for j in range(T_k):
            allowed = j <= q_offset + i
            sq = Square(side_length=CELL, stroke_width=1, stroke_color=GREY_B)
            if not allowed:
                sq.set_fill(GREY_B, opacity=0.25)
            elif j < cached:
                sq.set_fill(BLUE, opacity=0.55)  # key came from the cache
            else:
                sq.set_fill(GREEN, opacity=0.7)  # key computed this step
            sq.move_to(RIGHT * j * CELL + DOWN * i * CELL)
            g.add(sq)
    return g


class KVCacheDecode(Scene):
    def construct(self):
        title = Text("Causal attention with a KV cache: prefill, then one query per decode step", font_size=22)
        title.to_edge(UP)
        self.add(title)
        panels = VGroup()
        specs = [
            ("prefill: 4 prompt tokens\nqueries 0-3 vs keys 0-3", 4, 4, 0, 0),
            ("decode step 1: token 4\n1 query vs 5 keys (4 cached)", 1, 5, 4, 4),
            ("decode step 2: token 5\n1 query vs 6 keys (5 cached)", 1, 6, 5, 5),
        ]
        for label, T_q, T_k, off, cached in specs:
            g = grid(T_q, T_k, off, cached)
            cap = Text(label, font_size=16, line_spacing=1.1).next_to(g, DOWN, buff=0.35)
            panels.add(VGroup(g, cap))
        panels.arrange(RIGHT, buff=1.1, aligned_edge=UP).next_to(title, DOWN, buff=0.8)
        self.add(panels)
        legend = VGroup(
            Square(side_length=0.25, stroke_width=1).set_fill(GREEN, opacity=0.7),
            Text("K/V computed this step", font_size=16),
            Square(side_length=0.25, stroke_width=1).set_fill(BLUE, opacity=0.55),
            Text("K/V read from cache (B, H, T_cache, d_head)", font_size=16),
            Square(side_length=0.25, stroke_width=1).set_fill(GREY_B, opacity=0.25),
            Text("masked (j > i): -inf before softmax", font_size=16),
        ).arrange_in_grid(rows=3, cols=2, buff=0.2, col_alignments="ll").to_edge(DOWN, buff=0.4)
        self.add(legend)
        note = Text("per step: project 1 token, attend to T_cache + 1 keys -> O(T) instead of O(T^2)", font_size=16, color=YELLOW)
        note.next_to(legend, UP, buff=0.3)
        self.add(note)
        self.wait(0.1)
