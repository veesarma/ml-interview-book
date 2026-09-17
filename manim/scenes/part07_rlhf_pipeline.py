"""Manim still of the RLHF training loop: prompt -> policy samples -> reward model scores
-> PPO update, with the reference model's KL penalty and the critic. Uses Text only (no LaTeX).

Render:  manim -qm -s --format=png manim/scenes/part07_rlhf_pipeline.py RLHFPipeline
then copy media/images/part07_rlhf_pipeline/RLHFPipeline_ManimCE_v0.21.0.png to
docs/assets/figures/part07_rlhf_pipeline_manim.png
"""
from manim import (
    BLUE, GREEN, ORANGE, PURPLE, RED, WHITE, DOWN, LEFT, RIGHT, UP,
    Arrow, Line, Rectangle, Scene, Text, VGroup, config,
)

config.background_color = WHITE

GREY = "#666666"
INK = "#333333"


def box(label: str, sub: str, color, width: float = 3.0, height: float = 1.15) -> VGroup:
    """A titled box with a one-line caption underneath the title."""
    rect = Rectangle(width=width, height=height, color=color,
                     fill_color=color, fill_opacity=0.10, stroke_width=2.5)
    title = Text(label, font_size=21, color=color, weight="BOLD")
    body = Text(sub, font_size=14, color=INK)
    title.move_to(rect.get_center() + UP * 0.24)
    body.move_to(rect.get_center() + DOWN * 0.24)
    return VGroup(rect, title, body)


class RLHFPipeline(Scene):
    def construct(self) -> None:
        heading = Text("One RLHF (PPO) iteration for a language model",
                       font_size=32, color="#111111")
        heading.to_edge(UP, buff=0.45)
        self.add(heading)

        prompt = box("Prompt batch x", "sampled from the prompt set", GREY, width=3.1)
        policy = box("Policy  pi_theta", "samples y = (a_1 ... a_T)", BLUE, width=3.1)
        rm = box("Reward model  r_phi", "scalar score at the last token", GREEN, width=3.4)
        ref = box("Reference  pi_ref", "frozen SFT model", PURPLE, width=3.1)
        critic = box("Critic  V_psi", "per-token value V(s_t)", ORANGE, width=3.1)
        update = box("PPO update", "L_clip + c_v L_value - c_e H", RED, width=3.4)

        # Two rows, well separated, with the update box on the right of the lower row.
        top_y, bot_y = 1.45, -1.35
        prompt.move_to(LEFT * 4.85 + UP * top_y)
        policy.move_to(LEFT * 1.05 + UP * top_y)
        rm.move_to(RIGHT * 2.95 + UP * top_y)
        ref.move_to(LEFT * 4.85 + UP * bot_y)
        critic.move_to(LEFT * 1.05 + UP * bot_y)
        update.move_to(RIGHT * 2.95 + UP * bot_y)

        def labelled(a, b, text, where, buff=0.12, size=14):
            arrow = Arrow(a, b, buff=0.08, color=INK, stroke_width=3, max_tip_length_to_length_ratio=0.16)
            lab = Text(text, font_size=size, color=INK).next_to(arrow, where, buff=buff)
            return VGroup(arrow, lab)

        flows = VGroup(
            labelled(prompt.get_right(), policy.get_left(), "x", UP, buff=0.08),
            labelled(policy.get_right(), rm.get_left(), "(x, y)", UP, buff=0.08),
            labelled(policy.get_bottom(), ref.get_top() + RIGHT * 1.0,
                     "log pi_ref(a_t | s_t)", LEFT, buff=0.10),
            labelled(ref.get_right(), critic.get_left(), "R_t", UP, buff=0.08),
            labelled(critic.get_right(), update.get_left(), "A_t via GAE", UP, buff=0.08),
            labelled(rm.get_bottom(), update.get_top(), "R_T = r_phi(x, y)", RIGHT, buff=0.10),
        )

        # The per-token KL reward, stated once under the lower row rather than on an arrow.
        kl_note = Text("R_t = -beta (log pi_theta - log pi_ref) for every response token, "
                       "plus r_phi(x, y) at the last one",
                       font_size=15, color=PURPLE)
        kl_note.next_to(ref, DOWN, buff=0.34, aligned_edge=LEFT)

        # Feedback routed out to the right of every box, up to a clear lane, then back left.
        lane_y, lane_x = 2.62, 5.55
        update_y = update.get_center()[1]
        policy_x = policy.get_center()[0]
        back = VGroup(
            Line(update.get_right(), [lane_x, update_y, 0], color=RED, stroke_width=3),
            Line([lane_x, update_y, 0], [lane_x, lane_y, 0], color=RED, stroke_width=3),
            Line([lane_x, lane_y, 0], [policy_x, lane_y, 0], color=RED, stroke_width=3),
            Arrow([policy_x, lane_y, 0], policy.get_top(), buff=0.02, color=RED,
                  stroke_width=3, max_tip_length_to_length_ratio=0.35),
        )
        across = back[2]
        back_label = Text("new theta (per-token ratio clipped to 1 +/- eps)",
                          font_size=15, color=RED)
        back_label.next_to(across, UP, buff=0.10)

        foot = VGroup(
            Text("Four networks in memory: policy and critic are trained; reference and reward model are frozen.",
                 font_size=15, color=INK),
            Text("Generation dominates wall-clock, so rollouts usually run on a separate inference engine.",
                 font_size=15, color=INK),
        ).arrange(DOWN, buff=0.10)
        foot.next_to(kl_note, DOWN, buff=0.34)

        self.add(prompt, policy, rm, ref, critic, update, flows, kl_note, back, back_label, foot)
