"""The canonical VLM: image -> vision encoder -> projector -> LLM, with visual-token counts per design.

Writes docs/assets/figures/part08_vlm_architecture.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part08_vlm_architecture.png"


def box(ax, x, y, w, h, text, color, fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="black", linewidth=1.2))


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), facecolor="white", gridspec_kw={"height_ratios": [1.15, 1]})
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.2, 2.2, 1.8, 1.2, "image\n(3, 336, 336)", colors[7])
    box(ax, 2.6, 2.2, 2.4, 1.2, "vision encoder\nViT-L/14 (CLIP/SigLIP)\n→ (576, 1024)", colors[0])
    box(ax, 5.6, 2.2, 2.0, 1.2, "projector\nMLP / resampler\n→ (N_v, d_llm)", colors[1])
    box(ax, 8.2, 2.2, 3.6, 1.2, "LLM (decoder)\n[text ‖ N_v visual tokens ‖ text]\ncausal attention over all", colors[2])
    arrow(ax, 2.0, 2.8, 2.6, 2.8); arrow(ax, 5.0, 2.8, 5.6, 2.8); arrow(ax, 7.6, 2.8, 8.2, 2.8)
    box(ax, 0.2, 0.4, 4.8, 1.2, "text: 'USER: <image>\\nWhat is in the picture? ASSISTANT:'\n<image> is ONE placeholder id, expanded to N_v embeddings", colors[3], fontsize=8.5)
    arrow(ax, 5.0, 1.0, 8.6, 2.2)
    ax.text(0.2, 3.75, "Stage 1: train projector only (caption pairs).  Stage 2: projector + LLM (instruction data).  Stage 3 (optional): unfreeze vision encoder.", fontsize=8.5)
    ax.set_title("LLaVA-style VLM: the visual tokens are ordinary positions in the LLM sequence", fontsize=10, loc="left")

    designs = [
        ("ViT-L/14 @ 224, no compression", 256),
        ("ViT-L/14 @ 336 (LLaVA-1.5)", 576),
        ("336 AnyRes 2×2 tiles + global (LLaVA-NeXT)", 5 * 576),
        ("448², P=14, 2×2 pixel-shuffle (InternVL-style)", 256),
        ("Perceiver resampler, 64 queries (Flamingo-style)", 64),
        ("gated cross-attention (Flamingo / Llama 3.2-Vision)", 0),
    ]
    names = [d[0] for d in designs]
    vals = [d[1] for d in designs]
    y = range(len(designs))[::-1]
    ax2.barh(list(y), vals, color=[colors[i % 10] for i in range(len(designs))], alpha=0.85)
    for yi, v in zip(y, vals):
        ax2.text(v + 30, yi, f"{v} tokens" if v else "0 in-sequence tokens (KV cache unchanged)", va="center", fontsize=8)
    ax2.set_yticks(list(y)); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlim(0, 3600); ax2.set_xlabel("visual tokens per image entering the LLM sequence (prefill FLOPs and KV cache scale with this)", fontsize=8.5)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
