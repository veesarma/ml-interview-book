"""Dependency graph of the 60-item coding canon, grouped by area.

Writes docs/assets/figures/part16_canon_graph.png.
Run from the repository root: python figures/part16_canon_graph.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part16_canon_graph.png"

# name, items label, (x, y) centre, colour key
CLUSTERS = [
    ("Maths + classical ML", "#1-#8", (0.5, 4.7), "found"),
    ("Optimizers", "#9, #10", (0.5, 3.4), "found"),
    ("Nets + autograd", "#11, #12", (2.4, 4.1), "nets"),
    ("Normalization", "#13-#15", (2.4, 2.8), "nets"),
    ("Convolution + CNN", "#16, #17", (4.3, 6.1), "vision"),
    ("Detection", "#18-#21", (6.2, 6.1), "vision"),
    ("ViT, DETR, CLIP", "#39-#41", (8.1, 6.1), "mm"),
    ("VLM projector", "#42", (10.0, 6.1), "mm"),
    ("RNN + LSTM", "#22, #23", (4.3, 4.7), "seq"),
    ("Tokenizer", "#24", (4.3, 3.5), "seq"),
    ("Attention core", "#25-#28", (6.2, 4.1), "seq"),
    ("Blocks + GPT", "#29-#32", (8.1, 4.1), "seq"),
    ("KV cache + RoPE", "#33, #34", (10.0, 5.0), "llm"),
    ("GQA, SwiGLU, MoE", "#35-#37", (10.0, 3.9), "llm"),
    ("LoRA", "#38", (10.0, 2.8), "llm"),
    ("RL: DP to PPO", "#47-#53", (6.2, 2.2), "rl"),
    ("RM, DPO, GRPO", "#54-#56", (10.0, 1.7), "post"),
    ("Retrieval + RAG", "#45, #46", (4.3, 1.5), "ret"),
    ("Eval harness", "#59", (6.2, 0.9), "ret"),
    ("VAE + DDPM", "#43, #44", (4.3, 0.3), "gen"),
    ("DDP + TP, int8", "#57, #58", (10.0, 0.6), "sys"),
    ("Capstone agent", "#60", (12.3, 3.9), "cap"),
]

# (source, destination, arc curvature)
EDGES = [
    ("Maths + classical ML", "Nets + autograd", 0.0),
    ("Optimizers", "Nets + autograd", 0.0),
    ("Nets + autograd", "Normalization", 0.0),
    ("Nets + autograd", "Convolution + CNN", 0.0),
    ("Nets + autograd", "RNN + LSTM", 0.0),
    ("Normalization", "Attention core", -0.05),
    ("Normalization", "VAE + DDPM", 0.25),
    ("Maths + classical ML", "Retrieval + RAG", 0.28),
    ("Convolution + CNN", "Detection", 0.0),
    ("Detection", "ViT, DETR, CLIP", 0.0),
    ("ViT, DETR, CLIP", "VLM projector", 0.0),
    ("RNN + LSTM", "Attention core", 0.0),
    ("Tokenizer", "Attention core", 0.0),
    ("Attention core", "Blocks + GPT", 0.0),
    ("Attention core", "ViT, DETR, CLIP", -0.18),
    ("Blocks + GPT", "KV cache + RoPE", 0.0),
    ("Blocks + GPT", "GQA, SwiGLU, MoE", 0.0),
    ("Blocks + GPT", "LoRA", 0.0),
    ("Blocks + GPT", "DDP + TP, int8", 0.30),
    ("Blocks + GPT", "RM, DPO, GRPO", -0.22),
    ("RL: DP to PPO", "RM, DPO, GRPO", 0.10),
    ("Retrieval + RAG", "Eval harness", 0.0),
    ("VLM projector", "Capstone agent", 0.0),
    ("KV cache + RoPE", "Capstone agent", 0.0),
    ("RM, DPO, GRPO", "Capstone agent", 0.0),
    ("Eval harness", "Capstone agent", -0.36),
]

COLOURS = {
    "found": "#1f77b4", "nets": "#ff7f0e", "vision": "#2ca02c", "seq": "#d62728",
    "llm": "#9467bd", "mm": "#8c564b", "gen": "#e377c2", "ret": "#7f7f7f",
    "rl": "#bcbd22", "post": "#17becf", "sys": "#4c72b0", "cap": "#111111",
}

BOX_W, BOX_H = 1.62, 0.74


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=150)
    fig.patch.set_facecolor("white")
    centres = {name: xy for name, _, xy, _ in CLUSTERS}

    for src, dst, rad in EDGES:
        x0, y0 = centres[src]
        x1, y1 = centres[dst]
        ax.add_patch(FancyArrowPatch(
            (x0 + BOX_W / 2, y0), (x1 - BOX_W / 2, y1),
            arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
            color="#8c8c8c", connectionstyle=f"arc3,rad={rad}", zorder=1,
        ))

    for name, items, (x, y), key in CLUSTERS:
        colour = COLOURS[key]
        r, g, b = to_rgb(colour)
        tint = (1 - 0.13 * (1 - r), 1 - 0.13 * (1 - g), 1 - 0.13 * (1 - b))
        ax.add_patch(FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.6, edgecolor=colour, facecolor=tint, zorder=2,
        ))
        ax.text(x, y + 0.13, name, ha="center", va="center", fontsize=8.5, zorder=3)
        ax.text(x, y - 0.17, items, ha="center", va="center", fontsize=8.5,
                color=colour, fontweight="bold", zorder=3, family="monospace")

    ax.set_xlim(-0.55, 13.4)
    ax.set_ylim(-0.45, 6.9)
    ax.axis("off")
    ax.set_title("The coding canon: what unlocks what", fontsize=13, pad=12)
    ax.text(-0.5, -0.32,
            "Arrows point from a prerequisite to what it unlocks. Anything with no incoming arrow "
            "can be written on day one.",
            fontsize=8.5, color="#444444", ha="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
