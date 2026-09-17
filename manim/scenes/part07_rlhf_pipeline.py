"""Manim still of the RLHF training loop: prompt -> policy samples -> reward model scores
-> PPO update, with the reference model's KL penalty and the critic. Uses Text only (no LaTeX).

Render:  manim -qm -s --format=png manim/scenes/part07_rlhf_pipeline.py RLHFPipeline
then copy media/images/part07_rlhf_pipeline/RLHFPipeline_ManimCE_v0.21.0.png to
docs/assets/figures/part07_rlhf_pipeline_manim.png
"""
from manim import (
    BLUE, GREEN, ORANGE, PURPLE, RED, WHITE, YELLOW, DOWN, LEFT, RIGHT, UP,
    Arrow, Rectangle, Scene, Text, VGroup, config,
)

config.background_color = WHITE


def box(label: str, sub: str, color, width: float = 2.6, height: float = 1.1) -> VGroup:
    rect = Rectangle(width=width, height=height, color=color, fill_color=color, fill_opacity=0.12, stroke_width=2)
    title = Text(label, font_size=24, color=color, weight="BOLD")
    body = Text(sub, font_size=16, color="#333333")
    title.move_to(rect.get_center() + UP * 0.25)
    body.move_to(rect.get_center() + DOWN * 0.22)
    return VGroup(rect, title, body)


class RLHFPipeline(Scene):
    def construct(self) -> None:
        heading = Text("One RLHF (PPO) iteration for a language model", font_size=30, color="#111111")
        heading.to_edge(UP, buff=0.35)
        self.add(heading)

        prompt = box("Prompt batch x", "sampled from the prompt set", "#555555")
        policy = box("Policy  pi_theta", "samples y = (a_1 ... a_T)", BLUE)
        rm = box("Reward model  r_phi", "scalar score at last token", GREEN)
        ref = box("Reference  pi_ref", "frozen SFT model", PURPLE)
        critic = box("Critic  V_psi", "per-token value V(s_t)", ORANGE)
        update = box("PPO update", "L_clip + c_v L_value - c_e H", RED, width=3.0)

        prompt.move_to(LEFT * 5.0 + UP * 1.2)
        policy.move_to(LEFT * 1.6 + UP * 1.2)
        rm.move_to(RIGHT * 1.9 + UP * 1.2)
        ref.move_to(LEFT * 1.6 + DOWN * 1.2)
        critic.move_to(RIGHT * 1.9 + DOWN * 1.2)
        update.move_to(RIGHT * 5.2 + UP * 0.0)

        arrows = VGroup(
            Arrow(prompt.get_right(), policy.get_left(), buff=0.05, color="#333333"),
            Arrow(policy.get_right(), rm.get_left(), buff=0.05, color="#333333"),
            Arrow(policy.get_bottom(), ref.get_top(), buff=0.05, color="#333333"),
            Arrow(rm.get_right(), update.get_left() + UP * 0.3, buff=0.05, color="#333333"),
            Arrow(ref.get_right(), critic.get_left(), buff=0.05, color="#333333"),
            Arrow(critic.get_right(), update.get_left() + DOWN * 0.3, buff=0.05, color="#333333"),
        )
        labels = VGroup(
            Text("(x, y)", font_size=16, color="#333333").next_to(arrows[1], UP, buff=0.05),
            Text("log pi_ref(a_t | s_t)", font_size=16, color="#333333").next_to(arrows[2], RIGHT, buff=0.08),
            Text("R_T = r_phi(x, y)", font_size=16, color="#333333").next_to(arrows[3], UP, buff=0.05),
            Text("R_t -= beta * (log pi - log pi_ref)", font_size=16, color="#333333").next_to(arrows[4], DOWN, buff=0.05),
            Text("A_t via GAE", font_size=16, color="#333333").next_to(arrows[5], DOWN, buff=0.05),
        )
        back = Arrow(update.get_top(), policy.get_top() + UP * 0.05, buff=0.05, color=RED, path_arc=-1.2)
        back_label = Text("new theta (ratio clipped to 1 +/- eps)", font_size=16, color=RED)
        back_label.next_to(heading, DOWN, buff=0.15)

        self.add(prompt, policy, rm, ref, critic, update, arrows, labels, back, back_label)
        foot = Text("Four networks live in memory: policy, reference, reward model, critic. Generation dominates wall-clock.",
                    font_size=18, color="#333333")
        foot.to_edge(DOWN, buff=0.3)
        self.add(foot)
