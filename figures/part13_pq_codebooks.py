"""Product quantisation: two 2-D sub-spaces of a 4-D vector, each with its own
k-means codebook. A vector is stored as (code_1, code_2).

Writes docs/assets/figures/part13_pq_codebooks.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.retrieval.pq import ProductQuantizer  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_pq_codebooks.png"


def main() -> None:
    rng = np.random.default_rng(0)
    N = 600
    X = np.concatenate([rng.normal(size=(N, 2)) * [1.0, 0.4] + [c, 0] for c in (-3, 0, 3)])  # (3N, 2)
    Y = rng.normal(size=(3 * N, 2)) @ np.array([[1.0, 0.8], [0.0, 0.6]])  # (3N, 2) correlated
    V = np.concatenate([X, Y], axis=1)  # (3N, 4)
    pq = ProductQuantizer(n_subvectors=2, n_codes=8, n_iters=20).train(V)
    codes = pq.encode(V)  # (3N, 2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for m, ax in enumerate(axes):
        sub = V[:, 2 * m : 2 * m + 2]
        ax.scatter(sub[:, 0], sub[:, 1], c=codes[:, m], cmap="tab10", s=6, alpha=0.6)
        C = pq.codebooks[m]  # (K, 2)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=120, color="k", zorder=3, label="codewords")
        for k, c in enumerate(C):
            ax.annotate(str(k), c, textcoords="offset points", xytext=(5, 5))
        ax.set_title(f"sub-space {m + 1}: dims {2*m}-{2*m+1}, K=8 codewords")
        ax.legend(loc="upper left")
    v0 = V[0]
    fig.suptitle(f"vector {np.round(v0, 2).tolist()}  ->  codes {codes[0].tolist()}  (16 bytes -> 2 x 3 bits)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
