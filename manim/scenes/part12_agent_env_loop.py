"""Agent-environment loop with the Bellman backup written on it (still image).

Render:  manim -qm -s --format=png manim/scenes/part12_agent_env_loop.py AgentEnvLoop
Copy the PNG to docs/assets/figures/part12_agent_env_loop.png.
Uses ``Text`` only (no LaTeX in the build environment).
"""
from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Create,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
)


class AgentEnvLoop(Scene):
    def construct(self) -> None:
        self.camera.background_color = WHITE
        agent_box = RoundedRectangle(corner_radius=0.2, width=3.6, height=1.6, color=BLUE, fill_opacity=0.12).shift(LEFT * 3.5 + UP * 1.2)
        agent = VGroup(Text("Agent", color=BLUE, font_size=30, weight="BOLD"), Text("policy  π(a | s)", color=BLUE, font_size=22)).arrange(DOWN, buff=0.15).move_to(agent_box)
        env_box = RoundedRectangle(corner_radius=0.2, width=3.6, height=1.6, color=GREEN, fill_opacity=0.12).shift(RIGHT * 3.5 + UP * 1.2)
        env = VGroup(Text("Environment", color=GREEN, font_size=30, weight="BOLD"), Text("P(s' | s, a),  R(s, a)", color=GREEN, font_size=22)).arrange(DOWN, buff=0.15).move_to(env_box)

        top = Arrow(agent_box.get_right() + UP * 0.35, env_box.get_left() + UP * 0.35, color=BLUE, buff=0.05)
        top_label = Text("action a_t", color=BLUE, font_size=22).next_to(top, UP, buff=0.1)
        bottom = Arrow(env_box.get_left() + DOWN * 0.35, agent_box.get_right() + DOWN * 0.35, color=GREEN, buff=0.05)
        bottom_label = Text("state s_{t+1},  reward r_{t+1}", color=GREEN, font_size=22).next_to(bottom, DOWN, buff=0.1)

        goal = Text("Goal:  maximise  E[ Σ_t γ^t r_{t+1} ]", color="#222222", font_size=26).shift(DOWN * 0.9)
        bellman = VGroup(
            Text("Bellman expectation:  V(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_s' P(s'|s,a) V(s') ]", color="#222222", font_size=22),
            Text("Bellman optimality:    V*(s) = max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V*(s') ]", color=YELLOW.darker() if hasattr(YELLOW, "darker") else "#8a6d00", font_size=22),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).shift(DOWN * 2.2)

        self.play(Create(agent_box), Create(env_box))
        self.add(agent, env, top, top_label, bottom, bottom_label, goal, bellman)
        self.wait(0.5)
