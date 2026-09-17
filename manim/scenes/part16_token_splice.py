"""Manim still of the VLM token splice: visual tokens in front of text tokens, with
the attention mask, the position ids, and the off-by-one that aligns the loss.
Uses Text only (no LaTeX).

Render:  manim -qm -s --format=png manim/scenes/part16_token_splice.py TokenSplice
then copy media/images/part16_token_splice/TokenSplice_ManimCE_v0.21.0.png to
docs/assets/figures/part16_token_splice_manim.png
"""
from manim import (
    BLUE, GREEN, ORANGE, RED, DOWN, LEFT, RIGHT, UP,
    Arrow, Rectangle, Scene, Text, VGroup, config,
)

config.background_color = "#FFFFFF"

GREY = "#777777"
INK = "#333333"

N_Q = 4          # visual tokens drawn
TEXT = ["<bos>", "how", "many", "red", "answer", ":", "2", "<eos>"]
CELL_W, CELL_H = 1.06, 0.62


def cell(label: str, colour, fill: float = 0.12) -> VGroup:
    rect = Rectangle(width=CELL_W, height=CELL_H, color=colour,
                     fill_color=colour, fill_opacity=fill, stroke_width=2.0)
    text = Text(label, font_size=15, color=INK)
    if text.width > CELL_W - 0.16:
        text.scale((CELL_W - 0.16) / text.width)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def row(labels, colours, y: float) -> VGroup:
    cells = VGroup(*[cell(t, c) for t, c in zip(labels, colours)])
    cells.arrange(RIGHT, buff=0.06)
    cells.move_to([0.0, y, 0.0])
    return cells


class TokenSplice(Scene):
    def construct(self) -> None:
        heading = Text("Splicing an image into a text sequence", font_size=31, color="#111111")
        heading.to_edge(UP, buff=0.40)
        self.add(heading)

        labels = [f"img {i}" for i in range(N_Q)] + TEXT
        colours = [BLUE] * N_Q + [GREEN] * len(TEXT)
        tokens = row(labels, colours, 1.35)
        self.add(tokens)

        total = N_Q + len(TEXT)
        pos = row([str(i) for i in range(total)], [GREY] * total, 0.60)
        for group in pos:
            group[0].set_stroke(opacity=0.35)
        self.add(pos)

        mask = row(["1"] * total, [ORANGE] * total, -0.12)
        for group in mask:
            group[0].set_stroke(opacity=0.35)
        self.add(mask)

        for text, target, colour in [
            ("inputs_embeds", tokens, INK),
            ("position_ids", pos, GREY),
            ("attention_mask", mask, "#b45f06"),
        ]:
            tag = Text(text, font_size=17, color=colour)
            tag.next_to(target, LEFT, buff=0.30)
            self.add(tag)

        vis_brace = Text(f"{N_Q} visual tokens, never padding", font_size=16, color=BLUE)
        vis_brace.next_to(tokens[:N_Q], UP, buff=0.22)
        self.add(vis_brace)

        txt_brace = Text("T text tokens", font_size=16, color=GREEN)
        txt_brace.next_to(tokens[N_Q:], UP, buff=0.22)
        self.add(txt_brace)

        # the off-by-one: logit at N_q + t - 1 scores text token t
        src = tokens[N_Q + 5]          # the ":" token, absolute index N_Q + 5
        dst = tokens[N_Q + 6]          # the answer token, absolute index N_Q + 6
        arrow = Arrow(src.get_bottom() + DOWN * 0.95, dst.get_bottom() + DOWN * 0.35,
                      color=RED, stroke_width=3.2, buff=0.05, max_tip_length_to_length_ratio=0.28)
        self.add(arrow)

        note = Text("logits[:, i] scores position i + 1", font_size=18, color=RED)
        note.next_to(mask, DOWN, buff=0.55)
        note.shift(RIGHT * 1.1)
        self.add(note)

        slice_note = Text(
            "so the logit for text token t sits at N_q + t - 1,\n"
            "and logits[:, N_q - 1 : N_q - 1 + T] undoes both offsets",
            font_size=18, color=INK, line_spacing=0.85,
        )
        slice_note.next_to(note, DOWN, buff=0.30)
        self.add(slice_note)

        loss_note = Text("loss is applied here only: the assistant turn",
                         font_size=16, color="#b45f06")
        loss_note.next_to(tokens[N_Q + 6: N_Q + 8], DOWN, buff=3.15)
        self.add(loss_note)
