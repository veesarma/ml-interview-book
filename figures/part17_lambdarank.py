"""Learning to rank: the RankNet pairwise cost and how LambdaRank re-weights pairs by |ΔNDCG|."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_lambdarank.png"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    d = np.linspace(-6, 6, 300)
    sigma = 1.0
    cost = np.log1p(np.exp(-sigma * d))
    grad = -sigma / (1 + np.exp(sigma * d))
    ax1.plot(d, cost, color="C0", lw=2, label="RankNet cost  log(1 + e^{-σ(s_i - s_j)})")
    ax1.plot(d, -grad, color="C1", lw=2, label="|λ_ij| = σ / (1 + e^{σ(s_i - s_j)})")
    ax1.set_xlabel("score margin  s_i - s_j   (i should rank above j)")
    ax1.set_ylabel("value")
    ax1.set_title("Pairwise cost and its gradient magnitude")
    ax1.legend(fontsize=8)

    # |ΔNDCG| for swapping a relevant (gain 1) doc at position i with a non-relevant doc at position j
    n = 10
    ranks = np.arange(1, n + 1)
    disc = 1.0 / np.log2(ranks + 1)
    idcg = disc[0]
    swap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            swap[i, j] = abs(disc[i] - disc[j]) / idcg
    im = ax2.imshow(swap, cmap="Blues", origin="upper")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(ranks)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(ranks)
    ax2.set_xlabel("position of the worse document j")
    ax2.set_ylabel("position of the better document i")
    ax2.set_title("|ΔNDCG| for swapping i and j: top positions dominate")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
