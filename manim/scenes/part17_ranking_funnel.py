"""Manim still: the multi-stage ranking funnel (retrieval -> pre-ranking -> ranking -> re-ranking).

Render: manim -qm -s --format=png manim/scenes/part17_ranking_funnel.py RankingFunnel
"""

from manim import (BLUE, DOWN, GREEN, LEFT, ORANGE, RIGHT, UP, WHITE, YELLOW, Arrow, Polygon, Scene, Text, VGroup)


class RankingFunnel(Scene):
    def construct(self) -> None:
        stages = [
            ("Corpus", "1B items", "inverted index / embedding index", BLUE),
            ("Retrieval", "~thousands", "two-tower ANN + heuristics, ~20 ms", GREEN),
            ("Pre-ranking", "~hundreds", "light DNN / distilled, ~15 ms", YELLOW),
            ("Ranking", "~tens", "multi-task DNN + sequence features, ~60 ms", ORANGE),
            ("Re-ranking", "top-10", "value model, diversity, policy, ~10 ms", WHITE),
        ]
        widths = [9.0, 7.0, 5.2, 3.5, 2.0]
        top = 3.2
        h = 1.15
        group = VGroup()
        for i, ((name, count, note, color), w) in enumerate(zip(stages, widths)):
            y0 = top - i * h
            w_next = widths[i + 1] if i + 1 < len(widths) else w * 0.7
            poly = Polygon([-w / 2, y0, 0], [w / 2, y0, 0], [w_next / 2, y0 - h + 0.08, 0], [-w_next / 2, y0 - h + 0.08, 0],
                           fill_color=color, fill_opacity=0.35, stroke_color=color, stroke_width=2)
            label = Text(f"{name}  ·  {count}", font_size=22).move_to([0, y0 - 0.38, 0])
            sub = Text(note, font_size=15).move_to([0, y0 - 0.75, 0])
            group.add(poly, label, sub)
        title = Text("The ranking funnel: cheap models see everything, expensive models see almost nothing", font_size=20)
        title.to_edge(UP, buff=0.15)
        self.add(title, group)
        arrow = Arrow(start=[5.6, top - 0.6, 0], end=[5.6, top - 4 * h + 0.2, 0], color=WHITE, stroke_width=3)
        cost = Text("cost per item grows", font_size=15).next_to(arrow, UP, buff=0.1)
        items = Text("items per query shrink", font_size=15).next_to(arrow, DOWN, buff=0.1)
        self.add(arrow, cost, items)
