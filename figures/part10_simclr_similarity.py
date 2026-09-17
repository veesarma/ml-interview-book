"""SimCLR: the 2B x 2B similarity matrix, the positives, and what temperature does to the loss.

Writes docs/assets/figures/part10_simclr_similarity.png.
Run:  python figures/part10_simclr_similarity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mlbook.ssl import simclr as S  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part10_simclr_similarity.png"
torch.set_num_threads(1)
B = 8


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    # 8 "images" as 16-dim identities; two views are the identity plus view-specific noise
    identity = F.normalize(torch.randn(B, 16), dim=1)
    z1 = identity + 0.35 * torch.randn(B, 16)
    z2 = identity + 0.35 * torch.randn(B, 16)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = (z @ z.t()).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor="white")

    im = axes[0].imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(2 * B):
        j = i + B if i < B else i - B
        axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="black", linewidth=1.4))
    axes[0].axhline(B - 0.5, color="black", linewidth=1.0)
    axes[0].axvline(B - 0.5, color="black", linewidth=1.0)
    axes[0].set_title("cosine similarity of the 2B embeddings\nboxed cells are the positives", fontsize=10)
    axes[0].set_xlabel("view index"); axes[0].set_ylabel("anchor index")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # what the softmax sees for one anchor, at two temperatures
    row = sim[0].copy()
    row[0] = -np.inf                                   # the anchor is excluded from its own denominator
    for t, c, ls in ((1.0, colors[0], "-"), (0.1, colors[1], "--")):
        p = np.exp(row / t - np.nanmax(row[np.isfinite(row)] / t))
        p[~np.isfinite(row)] = 0.0
        p = p / p.sum()
        axes[1].bar(np.arange(2 * B) + (0.2 if t == 0.1 else -0.2), p, width=0.4,
                    color=c, label=f"temperature {t}")
    axes[1].axvline(B, color="black", linestyle=":", linewidth=1.0)
    axes[1].text(B + 0.15, 0.85, "the positive", fontsize=8, color="#555555")
    axes[1].set_xlabel("candidate index"); axes[1].set_ylabel("softmax probability")
    axes[1].set_title("Anchor 0: a cold temperature concentrates\nall the gradient on the hardest negatives", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)

    # loss against batch size at fixed embedding quality, and the chance level
    sizes = [4, 8, 16, 32, 64, 128, 256]
    losses, chance = [], []
    for n in sizes:
        idt = F.normalize(torch.randn(n, 32), dim=1)
        a = idt + 0.35 * torch.randn(n, 32)
        b = idt + 0.35 * torch.randn(n, 32)
        losses.append(S.nt_xent_loss(a, b, 0.2).item())
        chance.append(np.log(2 * n - 1))
    axes[2].plot(sizes, losses, "o-", color=colors[0], label="NT-Xent loss")
    axes[2].plot(sizes, chance, "--", color="#888888", label="chance level log(2B - 1)")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("batch size B"); axes[2].set_ylabel("loss (nats)")
    axes[2].set_title("The loss value is not comparable across batch sizes", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].grid(alpha=0.25, linewidth=0.5)
    for ax in axes[1:]:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
