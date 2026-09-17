"""Why ByteDance built a collisionless embedding table (Monolith, 2022).

With a fixed-size hashed embedding table of M rows and N distinct IDs, the expected
fraction of IDs that share a row with at least one other ID is 1 - (1 - 1/M)^(N-1).
Monolith's paper argues these collisions hurt quality for sparse, fast-changing ID
features and replaces the fixed table with a cuckoo hashmap plus expiry.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_consumer_hash_collisions.png"


def main() -> None:
    load = np.logspace(-2, 1.3, 200)                     # (200,) N / M
    collide = 1.0 - np.exp(-load)                        # (200,) ≈ 1 - (1-1/M)^(N-1) for large M
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(load, collide, lw=2, color="C0")
    for x, txt in [(0.1, "N = M/10:\n~10% of IDs collide"), (1.0, "N = M:\n~63% collide"), (10.0, "N = 10M:\nnearly all collide")]:
        y = 1 - np.exp(-x)
        ax.plot([x], [y], "o", color="C3", ms=7, markeredgecolor="white")
        ax.annotate(txt, (x, y), xytext=(x, y - 0.3 if x > 0.5 else y + 0.25), fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("distinct IDs / table rows  (N / M)")
    ax.set_ylabel("expected fraction of IDs sharing a row")
    ax.set_ylim(0, 1.05)
    ax.set_title("Hashed embedding tables collide long before they are 'full'")
    ax.grid(alpha=0.25, which="both")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
