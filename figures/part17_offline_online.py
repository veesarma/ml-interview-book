"""Offline metric gains vs online A/B outcomes: why a +2% AUC can be flat online."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_offline_online.png"


def main() -> None:
    rng = np.random.default_rng(4)
    n = 60
    offline = rng.uniform(-0.5, 3.0, n)                       # % AUC change vs control
    online = 0.35 * offline + rng.normal(0, 0.6, n)           # % engagement change, noisy and attenuated
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(offline, online, color="C0", s=30, alpha=0.8, label="one launch candidate")
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    xs = np.linspace(-0.5, 3, 10)
    ax.plot(xs, 0.35 * xs, color="C1", lw=2, label="typical attenuation (illustrative)")
    ax.fill_between(xs, -1.2, 1.2, color="C7", alpha=0.12)
    ax.text(1.4, -1.05, "A/B noise band: cannot tell +0.5% from 0 without power", fontsize=8)
    ax.set_xlabel("offline improvement (% AUC / NDCG vs control)")
    ax.set_ylabel("online improvement (% engagement vs control)")
    ax.set_title("Offline gains are attenuated online; feedback loops and logging policy bias explain the rest")
    ax.legend(fontsize=8, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
