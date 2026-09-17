"""Schematic of an HNSW index: three stacked layers, decreasing density, and the
greedy search path from the top entry point down to the nearest neighbour.

Writes docs/assets/figures/part13_hnsw_layers.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.retrieval.hnsw import HNSW  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_hnsw_layers.png"


def main() -> None:
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 10, size=(120, 2))  # (N, 2)
    idx = HNSW(M=4, ef_construction=32, seed=1)
    for x in X:
        idx.add(x)
    q = np.array([7.5, 2.5])
    n_layers = len(idx.graph)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharex=True, sharey=True)
    if n_layers == 1:
        axes = [axes]
    # trace the greedy path: layer by layer
    ep = idx.entry
    path = {}
    for l in range(idx.levels[idx.entry], -1, -1):
        found = idx._search_layer(q, ep, 1 if l > 0 else 16, l)
        path[l] = (ep, found[0][1])
        ep = found[0][1]
    for col, l in enumerate(range(n_layers - 1, -1, -1)):
        ax = axes[col]
        nodes = list(idx.graph[l].keys())
        for i in nodes:
            for j in idx.graph[l][i]:
                if j > i:
                    ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color="0.8", lw=0.8, zorder=1)
        P = X[nodes]
        ax.scatter(P[:, 0], P[:, 1], s=18, color="C0", zorder=2)
        start, end = path[l]
        ax.annotate("", xy=X[end], xytext=X[start], arrowprops=dict(arrowstyle="->", color="C3", lw=2), zorder=4)
        ax.scatter(*X[start], s=80, facecolor="none", edgecolor="C3", lw=2, zorder=3)
        ax.scatter(*q, marker="*", s=200, color="C1", zorder=5, label="query")
        ax.set_title(f"layer {l}: {len(nodes)} nodes" + ("  (entry point)" if l == n_layers - 1 else ""))
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="upper left")
    fig.suptitle("HNSW: greedy descent through sparse upper layers, then a beam search at layer 0")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
