"""Manim still of the VLM token splice: visual tokens in front of text tokens, with
the position ids, the attention mask, and the row that shows which text token each
logit scores. Uses Text only (no LaTeX).

Render:  manim -qm -s --format=png manim/scenes/part16_token_splice.py TokenSplice
then copy media/images/part16_token_splice/TokenSplice_ManimCE_v0.21.0.png to
docs/assets/figures/part16_token_splice_manim.png
"""
from manim import (
    BLUE, GREEN, ORANGE, RED, DOWN, LEFT, RIGHT, UP,
    Rectangle, Scene, Text, VGroup, config,
)

config.background_color = "#FFFFFF"

GREY = "#888888"
INK = "#333333"
AMBER = "#b45f06"

N_Q = 4                                              # visual tokens drawn
TEXT = ["<bos>", "how", "many", "red", "answer", ":", "2", "<eos>"]
LABELS = [f"img {i}" for i in range(N_Q)] + TEXT
TOTAL = len(LABELS)
SUPERVISED = {N_Q + 6, N_Q + 7}                      # the answer token and <eos>

CELL_W, CELL_H = 0.85, 0.58
SHIFT_X = 1.35


def cell(label: str, colour, fill: float, stroke: float) -> VGroup:
    rect = Rectangle(width=CELL_W, height=CELL_H, color=colour,
                     fill_color=colour, fill_opacity=fill, stroke_width=2.0)
    rect.set_stroke(opacity=stroke)
    text = Text(label, font_size=14, color=INK)
    if text.width > CELL_W - 0.12:
        text.scale((CELL_W - 0.12) / text.width)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


def grid_row(labels, colours, fills, strokes, y: float) -> VGroup:
    cells = VGroup(*[cell(t, c, f, s) for t, c, f, s in zip(labels, colours, fills, strokes)])
    cells.arrange(RIGHT, buff=0.05)
    cells.move_to([SHIFT_X, y, 0.0])
    return cells


def tag(text: str, target, colour) -> Text:
    label = Text(text, font_size=16, color=colour)
    label.next_to(target, LEFT, buff=0.22)
    return label


class TokenSplice(Scene):
    def construct(self) -> None:
        heading = Text("Splicing an image into a text sequence", font_size=30, color="#111111")
        heading.to_edge(UP, buff=0.42)
        self.add(heading)

        # row 1: the spliced token sequence
        colours = [BLUE] * N_Q + [GREEN] * len(TEXT)
        tokens = grid_row(LABELS, colours, [0.13] * TOTAL, [1.0] * TOTAL, 1.55)
        self.add(tokens, tag("inputs_embeds", tokens, INK))

        vis_note = Text("N_q visual tokens", font_size=15, color=BLUE)
        vis_note.next_to(tokens[:N_Q], UP, buff=0.18)
        txt_note = Text("T text tokens", font_size=15, color=GREEN)
        txt_note.next_to(tokens[N_Q:], UP, buff=0.18)
        self.add(vis_note, txt_note)

        # row 2: position ids
        pos = grid_row([str(i) for i in range(TOTAL)], [GREY] * TOTAL,
                       [0.06] * TOTAL, [0.4] * TOTAL, 0.78)
        self.add(pos, tag("position_ids", pos, GREY))

        # row 3: attention mask, all ones because nothing here is padding
        mask = grid_row(["1"] * TOTAL, [ORANGE] * TOTAL, [0.08] * TOTAL, [0.4] * TOTAL, 0.06)
        self.add(mask, tag("attention_mask", mask, AMBER))

        # row 4: which token each logit scores, and where the loss is applied
        predicts, pred_colours, pred_fills, pred_strokes = [], [], [], []
        for i in range(TOTAL):
            supervised = (i + 1) in SUPERVISED
            predicts.append(LABELS[i + 1] if i + 1 < TOTAL else "")
            pred_colours.append(RED if supervised else GREY)
            pred_fills.append(0.16 if supervised else 0.04)
            pred_strokes.append(1.0 if supervised else 0.3)
        pred = grid_row(predicts, pred_colours, pred_fills, pred_strokes, -0.66)
        self.add(pred, tag("logits[:, i] scores", pred, RED))

        note = Text("logits[:, i] scores position i + 1, so the logit for text token t sits at N_q + t - 1",
                    font_size=19, color=RED)
        note.next_to(pred, DOWN, buff=0.52)
        self.add(note)

        slice_note = Text("logits[:, N_q - 1 : N_q - 1 + T] undoes the visual offset and the shift at once",
                          font_size=19, color=INK)
        slice_note.next_to(note, DOWN, buff=0.26)
        self.add(slice_note)

        loss_note = Text("the loss is applied at the two red cells only: the assistant turn",
                         font_size=19, color=AMBER)
        loss_note.next_to(slice_note, DOWN, buff=0.26)
        self.add(loss_note)
