"""The post-training pipeline as a box diagram: Pretraining -> SFT -> RM -> RL, with the
models each stage produces and consumes. Writes docs/assets/figures/part07_pipeline.png.
Run from the repository root:  python figures/part07_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part07_pipeline.png"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor=color, edgecolor="black", linewidth=1.0, alpha=0.9))
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.05, body, ha="center", va="center", fontsize=7.8, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 linewidth=1.2, color="black"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.04, label, ha="center", va="bottom", fontsize=7.5)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor="white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    w, h, y = 2.3, 1.55, 1.8
    box(ax, 0.2, y, w, h, "Pretraining", "next-token CE on\ntrillions of tokens\n$\\rightarrow$ base LM $\\pi_0$", colors[0])
    box(ax, 2.9, y, w, h, "SFT", "(x, y) demonstrations\nassistant-only CE\n$\\rightarrow$ $\\pi_{\\mathrm{SFT}}$ (= $\\pi_{\\mathrm{ref}}$)", colors[1])
    box(ax, 5.6, y, w, h, "Reward model", "(x, $y_w$, $y_l$) preferences\n$-\\log\\sigma(r_w - r_l)$\n$\\rightarrow$ $r_\\phi(x, y)$", colors[2])
    box(ax, 8.3, y, w, h, "RL", "max $E[r]$ $-\\beta\\,$KL($\\pi_\\theta\\|\\pi_{\\mathrm{ref}}$)\nPPO / GRPO / DPO\n$\\rightarrow$ $\\pi_\\theta$ (deployed)", colors[3])

    for x0 in (2.5, 5.2, 7.9):
        arrow(ax, x0, y + h / 2, x0 + 0.4, y + h / 2)

    # Data feeding each stage
    ax.text(1.35, 3.85, "web + code corpora", ha="center", fontsize=8, style="italic")
    ax.text(4.05, 3.85, "human / synthetic demos", ha="center", fontsize=8, style="italic")
    ax.text(6.75, 3.85, "human / AI rankings", ha="center", fontsize=8, style="italic")
    ax.text(9.45, 3.85, "prompts (+ verifiers)", ha="center", fontsize=8, style="italic")
    for xc in (1.35, 4.05, 6.75, 9.45):
        arrow(ax, xc, 3.75, xc, y + h + 0.05)

    # RL consumes SFT policy as reference and RM as reward
    arrow(ax, 4.05, y - 0.05, 9.1, y - 0.05, style="-")
    arrow(ax, 9.1, y - 0.05, 9.1, y - 0.02)
    ax.text(6.6, y - 0.35, "$\\pi_{\\mathrm{SFT}}$ initialises $\\pi_\\theta$ and is frozen as $\\pi_{\\mathrm{ref}}$ (KL anchor)",
            ha="center", fontsize=7.8)
    ax.text(5.5, 0.55, "DPO skips the explicit RM: it fits $\\pi_\\theta$ directly on preferences using the implicit reward "
                       "$\\beta\\log\\frac{\\pi_\\theta}{\\pi_{\\mathrm{ref}}}$;  RLVR replaces $r_\\phi$ with a programmatic verifier.",
            ha="center", fontsize=8)
    ax.text(5.5, 0.2, "Llama 2 / InstructGPT: SFT $\\rightarrow$ RM $\\rightarrow$ PPO.   Llama 3: SFT $\\rightarrow$ RM (rejection sampling) $\\rightarrow$ DPO, iterated.   "
                      "DeepSeek-R1: cold-start SFT $\\rightarrow$ GRPO with verifiers.", ha="center", fontsize=7.5, color="#333333")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
