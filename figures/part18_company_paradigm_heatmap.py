"""Companies x ML paradigms matrix for the Part XVIII index. The weights are an
editorial reading of each company's public engineering write-ups (0 = not a hiring
focus, 1 = present, 2 = important, 3 = defines the company). Single-hue sequential
colour map; every cell is also labelled so the figure does not rely on colour."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_company_paradigm_heatmap.png"

PARADIGMS = ["Perception /\nautonomy", "Ranking /\nrecs", "Fraud /\nanomaly", "LLM /\npost-training",
             "Systems /\ninfra", "Data\nengines"]
COMPANIES = [
    ("Tesla", [3, 0, 0, 1, 3, 3]),
    ("Waymo", [3, 0, 0, 2, 2, 3]),
    ("Zoox / Nuro / Aurora", [3, 0, 0, 1, 2, 2]),
    ("NVIDIA", [3, 1, 0, 3, 3, 2]),
    ("Meta", [1, 3, 2, 3, 3, 2]),
    ("Google / YouTube", [1, 3, 2, 3, 3, 2]),
    ("TikTok / ByteDance", [1, 3, 2, 2, 3, 2]),
    ("Netflix / Spotify", [0, 3, 1, 1, 2, 1]),
    ("Pinterest", [2, 3, 1, 1, 2, 1]),
    ("Uber / DoorDash", [1, 3, 3, 1, 2, 1]),
    ("Airbnb", [1, 3, 2, 1, 2, 1]),
    ("Amazon", [2, 3, 3, 3, 3, 2]),
    ("Apple", [3, 2, 1, 3, 3, 1]),
    ("Stripe / fintech", [0, 1, 3, 1, 2, 1]),
    ("OpenAI / Anthropic / DeepMind", [1, 0, 0, 3, 3, 3]),
    ("Scale AI", [2, 0, 0, 3, 1, 3]),
]


def main() -> None:
    names = [c[0] for c in COMPANIES]
    M = np.array([c[1] for c in COMPANIES], dtype=float)  # (16, 6)
    fig, ax = plt.subplots(figsize=(8.2, 8.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(PARADIGMS)))
    ax.set_xticklabels(PARADIGMS, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    labels = {0: "", 1: "•", 2: "••", 3: "•••"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            ax.text(j, i, labels[v], ha="center", va="center", fontsize=10,
                    color="white" if v >= 2 else "0.25")
    # thin white grid between cells
    ax.set_xticks(np.arange(-0.5, len(PARADIGMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.5, pad=0.02)
    cbar.ax.set_yticklabels(["not a focus", "present", "important", "defines the company"], fontsize=8)
    cbar.outline.set_visible(False)
    ax.set_title("Which paradigms each company interviews for (editorial weighting from public write-ups)",
                 fontsize=10, pad=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
