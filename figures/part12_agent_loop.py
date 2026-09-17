"""The observe -> reason -> act loop of an LLM agent, with the reliability layer around the tools.

Writes docs/assets/figures/part12_agent_loop.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_agent_loop.png"


def box(ax, x, y, w, h, text, color, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=color, edgecolor="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, text="", rad=0.0, color="#333333", dx=0.0, dy=0.18, va="bottom"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, ha="center", va=va, fontsize=7.5, color=color)


def main() -> None:
    c = [to_hex(x) for x in plt.rcParams["axes.prop_cycle"].by_key()["color"]]  # tab10 as #rrggbb
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor="white")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, 0.3, 3.6, 2.2, 1.0, "User task\n+ system prompt\n+ tool schemas", "#f0f0f0", 8.5)
    box(ax, 3.4, 3.4, 3.2, 1.4, "Policy π_θ (the LLM)\nreason → choose action\n\naction ∈ {tool call (name, args),\n final answer}", c[0] + "40", 8.5, True)
    box(ax, 7.6, 3.5, 2.6, 1.2, "Environment\ntools: search, code, retrieval,\nAPIs, file system, browser", c[2] + "40", 8.5, True)
    box(ax, 3.4, 1.2, 3.2, 1.1, "Context / memory\nmessage history (s_t = prompt + all\nprior actions and observations)", c[1] + "35", 8)
    box(ax, 7.6, 1.2, 2.6, 1.1, "Reliability layer\nschema validation · timeouts\nretries · sandbox · permissions", c[3] + "30", 8)
    box(ax, 0.3, 1.2, 2.2, 1.1, "Trajectory recorder\n(s, a, o)_t → eval by\ntask success, cost, steps", "#f0f0f0", 8)
    arrow(ax, 2.5, 4.1, 3.4, 4.1)
    arrow(ax, 6.6, 4.3, 7.6, 4.3, "act: tool call")
    arrow(ax, 7.6, 3.75, 6.6, 3.75, "observe: tool result", color=c[2], dy=-0.14, va="top")
    arrow(ax, 5.0, 3.4, 5.0, 2.3, "append to state", color=c[1], dx=1.05, dy=-0.04)
    arrow(ax, 5.0, 2.3, 5.0, 3.4, "", color=c[1])
    arrow(ax, 8.9, 3.5, 8.9, 2.3, "every call passes through", color=c[3], dy=-0.02)
    arrow(ax, 3.4, 1.75, 2.5, 1.75, "log", color="#777777")
    ax.text(5.0, 0.55, "Terminates on a final answer or the step budget. The step budget, sandbox and permissioning are the safety envelope (Part XV).", ha="center", fontsize=8, color="#555555")
    ax.text(5.0, 0.2, "Training signal for agentic RL: reward = verifier(task, final state) at the end of the trajectory, credit spread over the tokens of every action.", ha="center", fontsize=8, color="#555555")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
