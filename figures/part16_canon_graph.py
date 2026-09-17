"""Dependency graph of the 60-item coding canon, grouped by area.

Writes docs/assets/figures/part16_canon_graph.png.
Run from the repository root: python figures/part16_canon_graph.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part16_canon_graph.png"

# name, items label, (x, y) centre, colour key
CLUSTERS = [
    ("Maths + classical ML", "#1-#8", (0.5, 4.6), "found"),
    ("Optimizers", "#9, #10", (0.5, 3.2), "found"),
    ("Nets + autograd", "#11, #12", (2.3, 3.9), "nets"),
    ("Normalization", "#13-#15", (2.3, 2.6), "nets"),
    ("Convolution + CNN", "#16, #17", (4.1, 5.6), "vision"),
    ("Detection", "#18-#21", (5.9, 5.6), "vision"),
    ("RNN + LSTM", "#22, #23", (4.1, 4.3), "seq"),
    ("Tokenizer", "#24", (4.1, 3.1), "seq"),
    ("Attention core", "#25-#28", (5.9, 3.7), "seq"),
    ("Blocks + GPT", "#29-#32", (7.7, 3.7), "seq"),
    ("KV cache + RoPE", "#33, #34", (9.5, 4.6), "llm"),
    ("GQA, SwiGLU, MoE", "#35-#37", (9.5, 3.3), "llm"),
    ("LoRA", "#38", (9.5, 2.1), "llm"),
    ("ViT, DETR, CLIP", "#39-#41", (7.7, 5.9), "mm"),
    ("VLM projector", "#42", (9.5, 5.9), "mm"),
    ("VAE + DDPM", "#43, #44", (7.7, 1.2), "gen"),
    ("Retrieval + RAG", "#45, #46", (4.1, 1.2), "ret"),
    ("Eval harness", "#59", (5.9, 0.4), "ret"),
    ("RL: DP to PPO", "#47-#53", (5.9, 2.0), "rl"),
    ("RM, DPO, GRPO", "#54-#56", (11.3, 3.3), "post"),
    ("DDP + TP, int8", "#57, #58", (11.3, 1.6), "sys"),
    ("Capstone agent", "#60", (11.3, 5.2), "cap"),
]

EDGES = [
    ("Maths + classical ML", "Nets + autograd"),
    ("Optimizers", "Nets + autograd"),
    ("Nets + autograd", "Normalization"),
    ("Nets + autograd", "Convolution + CNN"),
    ("Nets + autograd", "RNN + LSTM"),
    ("Normalization", "Attention core"),
    ("Convolution + CNN", "Detection"),
    ("RNN + LSTM", "Attention core"),
    ("Tokenizer", "Blocks + GPT"),
    ("Attention core", "Blocks + GPT"),
    ("Blocks + GPT", "KV cache + RoPE"),
    ("Blocks + GPT", "GQA, SwiGLU, MoE"),
    ("Blocks + GPT", "LoRA"),
    ("Attention core", "ViT, DETR, CLIP"),
    ("Convolution + CNN", "ViT, DETR, CLIP"),
    ("Detection", "ViT, DETR, CLIP"),
    ("ViT, DETR, CLIP", "VLM projector"),
    ("Nets + autograd", "VAE + DDPM"),
    ("Maths + classical ML", "Retrieval + RAG"),
    ("Retrieval + RAG", "Eval harness"),
    ("RL: DP to PPO", "RM, DPO, GRPO"),
    ("Blocks + GPT", "RM, DPO, GRPO"),
    ("Blocks + GPT", "DDP + TP, int8"),
    ("VLM projector", "Capstone agent"),
    ("RM, DPO, GRPO", "Capstone agent"),
    ("KV cache + RoPE", "Capstone agent"),
    ("Eval harness", "Capstone agent"),
]

COLOURS = {
    "found": "#1f77b4", "nets": "#ff7f0e", "vision": "#2ca02c", "seq": "#d62728",
    "llm": "#9467bd", "mm": "#8c564b", "gen": "#e377c2", "ret": "#7f7f7f",
    "rl": "#bcbd22", "post": "#17becf", "sys": "#4c72b0", "cap": "#111111",
}

BOX_W, BOX_H = 1.55, 0.72


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=150)
    fig.patch.set_facecolor("white")
    centres = {name: xy for name, _, xy, _ in CLUSTERS}

    for src, dst in EDGES:
        x0, y0 = centres[src]
        x1, y1 = centres[dst]
        ax.add_patch(FancyArrowPatch(
            (x0 + BOX_W / 2 * (1 if x1 > x0 else -1 if x1 < x0 else 0), y0),
            (x1 - BOX_W / 2 * (1 if x1 > x0 else -1 if x1 < x0 else 0), y1),
            arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
            color="#999999", connectionstyle="arc3,rad=0.08", zorder=1,
        ))

    for name, items, (x, y), key in CLUSTERS:
        colour = COLOURS[key]
        ax.add_patch(FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.6, edgecolor=colour, facecolor=colour + "22", zorder=2,
        ))
        ax.text(x, y + 0.13, name, ha="center", va="center", fontsize=8.5, zorder=3)
        ax.text(x, y - 0.17, items, ha="center", va="center", fontsize=8.5,
                color=colour, fontweight="bold", zorder=3, family="monospace")

    ax.set_xlim(-0.6, 12.4)
    ax.set_ylim(-0.3, 6.7)
    ax.axis("off")
    ax.set_title("The coding canon: what unlocks what", fontsize=13, pad=12)
    ax.text(-0.5, -0.15,
            "Arrows point from a prerequisite to what it unlocks. Anything with no incoming arrow "
            "can be written on day one.",
            fontsize=8.5, color="#444444", ha="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
