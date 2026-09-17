"""Agent-environment loop with the Bellman backup written underneath (still image).

Render:  manim -qm -s --format=png manim/scenes/part12_agent_env_loop.py AgentEnvLoop
Copy the PNG to docs/assets/figures/part12_agent_env_loop_manim.png.
Uses ``Text`` only (no LaTeX in the build environment), so subscripts are written inline.
"""
from manim import BLUE, DOWN, GREEN, LEFT, RIGHT, UP, WHITE, Arrow, Create, RoundedRectangle, Scene, Text, VGroup


class AgentEnvLoop(Scene):
    def construct(self) -> None:
        self.camera.background_color = WHITE
        gold = "#9a7b00"

        agent_box = RoundedRectangle(corner_radius=0.2, width=3.8, height=1.5, color=BLUE, fill_opacity=0.12)
        agent_box.shift(LEFT * 3.3 + UP * 2.1)
        agent = VGroup(
            Text("Agent", color=BLUE, font_size=32, weight="BOLD"),
            Text("policy  pi(a | s)", color=BLUE, font_size=22),
        ).arrange(DOWN, buff=0.15).move_to(agent_box)

        env_box = RoundedRectangle(corner_radius=0.2, width=3.8, height=1.5, color=GREEN, fill_opacity=0.12)
        env_box.shift(RIGHT * 3.3 + UP * 2.1)
        env = VGroup(
            Text("Environment", color=GREEN, font_size=32, weight="BOLD"),
            Text("P(s' | s, a),   R(s, a)", color=GREEN, font_size=22),
        ).arrange(DOWN, buff=0.15).move_to(env_box)

        # Forward: action leaves the agent along the top.
        top = Arrow(agent_box.get_right(), env_box.get_left(), color=BLUE, buff=0.08)
        top_label = Text("action  a_t", color=BLUE, font_size=24).next_to(top, UP, buff=0.12)

        # Return: state and reward come back along a path drawn BELOW both boxes.
        down_env = Arrow(env_box.get_bottom(), env_box.get_bottom() + DOWN * 0.75, color=GREEN, buff=0.05)
        back = Arrow(
            env_box.get_bottom() + DOWN * 0.75 + RIGHT * 0.02,
            agent_box.get_bottom() + DOWN * 0.75,
            color=GREEN,
            buff=0.0,
        )
        up_agent = Arrow(agent_box.get_bottom() + DOWN * 0.75, agent_box.get_bottom(), color=GREEN, buff=0.05)
        back_label = Text("next state  s_t+1,   reward  r_t+1", color=GREEN, font_size=24).next_to(back, DOWN, buff=0.12)

        goal = Text("Goal:  maximise  E[ sum_t  gamma^t  r_t+1 ]", color="#222222", font_size=28).shift(DOWN * 1.35)
        bellman = VGroup(
            Text("Bellman expectation:   V(s) = sum_a pi(a|s) [ R(s,a) + gamma sum_s' P(s'|s,a) V(s') ]",
                 color="#222222", font_size=22),
            Text("Bellman optimality:      V*(s) =    max_a    [ R(s,a) + gamma sum_s' P(s'|s,a) V*(s') ]",
                 color=gold, font_size=22),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT).shift(DOWN * 2.6)

        self.play(Create(agent_box), Create(env_box))
        self.add(agent, env, top, top_label, down_env, back, up_agent, back_label, goal, bellman)
        self.wait(0.5)
