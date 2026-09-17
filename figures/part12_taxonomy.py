"""Algorithm taxonomy for Part XII: model-free vs model-based, value vs policy, on- vs off-policy.

Writes docs/assets/figures/part12_taxonomy.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_taxonomy.png"


def box(ax, x, y, w, h, text, color, fontsize=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor=color, edgecolor="#333333", linewidth=0.8, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color="black")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9))


def main() -> None:
    c = [to_hex(x) for x in plt.rcParams["axes.prop_cycle"].by_key()["color"]]  # tab10 as #rrggbb
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor="white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    box(ax, 4.2, 5.6, 2.6, 0.6, "RL algorithms", "#e8e8e8", 10, True)
    box(ax, 0.6, 4.4, 3.4, 0.6, "Model-free\n(learn V, Q or π from samples)", c[0] + "55", 8.5, True)
    box(ax, 7.0, 4.4, 3.4, 0.6, "Model-based\n(learn or know P, R; plan)", c[2] + "55", 8.5, True)
    arrow(ax, 5.0, 5.6, 2.3, 5.0)
    arrow(ax, 6.0, 5.6, 8.7, 5.0)

    box(ax, 0.2, 3.1, 2.0, 0.6, "Value-based\n(Q → greedy π)", c[0] + "33", 8, True)
    box(ax, 2.6, 3.1, 2.0, 0.6, "Policy-based /\nactor–critic", c[1] + "33", 8, True)
    arrow(ax, 1.8, 4.4, 1.2, 3.7)
    arrow(ax, 2.8, 4.4, 3.6, 3.7)

    box(ax, 0.05, 1.55, 1.15, 1.2, "Off-policy\n\nQ-learning\nDQN, Double\nDueling, Rainbow", c[3] + "22", 6.6)
    box(ax, 1.3, 1.55, 1.15, 1.2, "On-policy\n\nSARSA\nExpected SARSA\nMC control", c[4] + "22", 6.6)
    box(ax, 2.55, 1.55, 1.15, 1.2, "On-policy\n\nREINFORCE\nA2C / A3C\nTRPO, PPO", c[4] + "22", 6.6)
    box(ax, 3.8, 1.55, 1.15, 1.2, "Off-policy\n\nDDPG, TD3\nSAC\n(continuous)", c[3] + "22", 6.6)
    arrow(ax, 0.9, 3.1, 0.65, 2.75)
    arrow(ax, 1.5, 3.1, 1.85, 2.75)
    arrow(ax, 3.3, 3.1, 3.1, 2.75)
    arrow(ax, 3.9, 3.1, 4.35, 2.75)

    box(ax, 6.8, 3.1, 1.8, 0.6, "Known model\n(planning / DP)", c[2] + "33", 8, True)
    box(ax, 8.9, 3.1, 1.8, 0.6, "Learned model\n(Dyna, world models)", c[2] + "33", 8, True)
    arrow(ax, 8.2, 4.4, 7.7, 3.7)
    arrow(ax, 9.3, 4.4, 9.8, 3.7)
    box(ax, 6.6, 1.55, 2.1, 1.2, "Value iteration\nPolicy iteration\nMCTS (AlphaZero)\nLQR / MPC", c[2] + "18", 7.5)
    box(ax, 8.9, 1.55, 2.0, 1.2, "Dyna-Q\nPETS, MBPO\nDreamer\n(imagined rollouts)", c[2] + "18", 7.5)
    arrow(ax, 7.7, 3.1, 7.65, 2.75)
    arrow(ax, 9.8, 3.1, 9.9, 2.75)

    box(ax, 0.05, 0.25, 4.95, 0.8,
        "Offline RL (CQL, IQL): off-policy from a fixed dataset, no interaction\n"
        "Imitation (BC, DAgger, GAIL): supervised from expert (s, a) pairs — no reward",
        "#f4f4f4", 6.8)
    box(ax, 5.55, 0.25, 4.95, 0.8,
        "Bandits: one-step MDP (no state transitions) — ε-greedy, UCB, Thompson\n"
        "RLHF / GRPO for LLMs: on-policy policy gradient with a learned reward (Part VII)",
        "#f4f4f4", 6.8)
    ax.text(5.3, 0.02, "Axes: model-free vs model-based (top), value vs policy (middle), on- vs off-policy (leaves).",
            ha="center", va="bottom", fontsize=8, color="#555555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
