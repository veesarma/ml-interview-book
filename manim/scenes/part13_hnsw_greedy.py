"""HNSW greedy search still: three layers, entry point at the top, greedy hops to
the neighbour closest to the query, descend, repeat. Uses Text only (no LaTeX).

Render:  manim -qm -s --format=png manim/scenes/part13_hnsw_greedy.py HNSWGreedySearch
"""
from __future__ import annotations

import numpy as np
from manim import (BLUE, GREY_B, ORANGE, RED, WHITE, YELLOW, Arrow, Dot, Line, Scene, Star, Text, VGroup, DOWN, LEFT, RIGHT, UP)


class HNSWGreedySearch(Scene):
    def construct(self) -> None:
        rng = np.random.default_rng(4)
        pts = rng.uniform(-1, 1, size=(40, 2)) * np.array([4.5, 1.1])  # (N, 2)
        levels = np.floor(-np.log(rng.uniform(1e-9, 1, 40)) / np.log(4)).astype(int)
        levels[0] = 2
        query = np.array([3.4, -0.6])
        layer_y = {2: 2.6, 1: 0.0, 0: -2.6}
        colours = {2: RED, 1: ORANGE, 0: BLUE}
        title = Text("HNSW greedy search: descend, hop to the closest neighbour, repeat", font_size=26).to_edge(UP, buff=0.15)
        self.add(title)
        nn_prev = 0
        for l in (2, 1, 0):
            members = [i for i in range(40) if levels[i] >= l]
            grp = VGroup()
            for i in members:
                grp.add(Dot(point=[pts[i, 0], pts[i, 1] * 0.55 + layer_y[l], 0], radius=0.05, color=colours[l]))
            # k-nearest links inside the layer (M = 3)
            for i in members:
                d = [(np.linalg.norm(pts[i] - pts[j]), j) for j in members if j != i]
                for _, j in sorted(d)[:3]:
                    grp.add(Line([pts[i, 0], pts[i, 1] * 0.55 + layer_y[l], 0], [pts[j, 0], pts[j, 1] * 0.55 + layer_y[l], 0], stroke_width=0.8, color=GREY_B))
            # greedy path from nn_prev to closest member of the query
            cur = nn_prev if nn_prev in members else members[0]
            while True:
                d_cur = np.linalg.norm(pts[cur] - query)
                cands = sorted(members, key=lambda j: np.linalg.norm(pts[j] - pts[cur]))[1:4]
                best = min(cands, key=lambda j: np.linalg.norm(pts[j] - query))
                if np.linalg.norm(pts[best] - query) >= d_cur:
                    break
                grp.add(Arrow([pts[cur, 0], pts[cur, 1] * 0.55 + layer_y[l], 0], [pts[best, 0], pts[best, 1] * 0.55 + layer_y[l], 0], buff=0.05, color=YELLOW, stroke_width=4, max_tip_length_to_length_ratio=0.2))
                cur = best
            nn_prev = cur
            grp.add(Star(n=5, outer_radius=0.16, color=WHITE, fill_opacity=1).move_to([query[0], query[1] * 0.55 + layer_y[l], 0]))
            label = Text(f"layer {l}  ({len(members)} nodes)" + ("  entry point" if l == 2 else ""), font_size=20, color=colours[l]).next_to(grp, LEFT, buff=0.2)
            self.add(grp, label)
            if l > 0:
                self.add(Arrow([pts[cur, 0], pts[cur, 1] * 0.55 + layer_y[l], 0], [pts[cur, 0], pts[cur, 1] * 0.55 + layer_y[l - 1], 0], color=YELLOW, stroke_width=3, buff=0.08))
        legend = Text("star = query    yellow = greedy hops    ef = 1 on upper layers, ef >= k on layer 0", font_size=18).to_edge(DOWN, buff=0.15)
        self.add(legend)
