"""Three ways to build a multimodal model: shared embedding space, modality encoders + shared LLM, one token stream.

Writes docs/assets/figures/part08_multimodal_strategies.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part08_multimodal_strategies.png"


def box(ax, x, y, w, h, text, color, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color="black", linewidth=1.1))


def main() -> None:
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor="white")
    mods = [("image", c[0]), ("text", c[1]), ("audio", c[2]), ("depth", c[3])]

    ax = axes[0]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("A. shared embedding space\n(ImageBind, CLIP)", fontsize=9, loc="left")
    for i, (m, col) in enumerate(mods):
        box(ax, 0.2, 5 - 1.3 * i, 1.3, 0.9, m, col)
        box(ax, 1.9, 5 - 1.3 * i, 1.6, 0.9, f"{m} encoder", col)
        arrow(ax, (1.5, 5.45 - 1.3 * i), (1.9, 5.45 - 1.3 * i))
        arrow(ax, (3.5, 5.45 - 1.3 * i), (4.4, 3.4))
    box(ax, 4.2, 2.6, 1.6, 1.6, "one unit\nsphere\nℝ^d", c[4], fs=8)
    ax.text(0.2, 0.15, "contrastive pairs (x, text); retrieval, zero-shot;\nno generation, no reasoning", fontsize=7.5)

    ax = axes[1]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("B. modality encoders + shared LLM\n(LLaVA, Qwen-VL, Gemini as described)", fontsize=9, loc="left")
    for i, (m, col) in enumerate(mods[:3]):
        box(ax, 0.2, 5 - 1.5 * i, 1.2, 0.9, m, col)
        box(ax, 1.7, 5 - 1.5 * i, 1.4, 0.9, "encoder\n+ projector" if m != "text" else "tokenizer", col, fs=7.5)
        arrow(ax, (1.4, 5.45 - 1.5 * i), (1.7, 5.45 - 1.5 * i))
        arrow(ax, (3.1, 5.45 - 1.5 * i), (3.7, 3.4))
    box(ax, 3.7, 1.6, 2.1, 3.4, "LLM\n(text out;\ncontinuous\nvisual tokens in)", c[4])
    ax.text(0.2, 0.15, "understanding + reasoning; generation only in text\n(image out needs a separate decoder)", fontsize=7.5)

    ax = axes[2]
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("C. everything tokenised\n(Chameleon, Unified-IO 2, Emu)", fontsize=9, loc="left")
    for i, (m, col) in enumerate(mods[:3]):
        box(ax, 0.2, 5 - 1.5 * i, 1.2, 0.9, m, col)
        box(ax, 1.7, 5 - 1.5 * i, 1.4, 0.9, "VQ tokenizer" if m != "text" else "BPE", col, fs=7.5)
        arrow(ax, (1.4, 5.45 - 1.5 * i), (1.7, 5.45 - 1.5 * i))
        arrow(ax, (3.1, 5.45 - 1.5 * i), (3.7, 3.4))
    box(ax, 3.7, 1.6, 2.1, 3.4, "one Transformer,\none discrete\nvocabulary;\nany-to-any\nnext-token", c[4])
    ax.text(0.2, 0.15, "generate any modality; pays VQ reconstruction loss\nand long sequences (1024+ tokens / image)", fontsize=7.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
