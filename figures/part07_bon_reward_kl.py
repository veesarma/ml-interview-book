"""Best-of-N: expected max of N standard-normal proxy rewards (Monte Carlo) against the KL
cost bound log N - (N-1)/N. Writes docs/assets/figures/part07_bon_reward_kl.png.
Run from the repository root:  python figures/part07_bon_reward_kl.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part07_bon_reward_kl.png"


def main() -> None:
    rng = np.random.default_rng(0)
    ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    expected_max = np.array([rng.standard_normal((20000, n)).max(axis=1).mean() for n in ns])
    kl = np.log(ns) - (ns - 1) / ns
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor="white")
    ax = axes[0]
    ax.plot(ns, expected_max, marker="o", color=colors[0], label="E[max of N]  (proxy reward, N(0,1))")
    ax.plot(ns, kl, marker="s", color=colors[1], label="KL bound  $\\log N - (N-1)/N$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N samples")
    ax.set_title("Proxy reward grows like $\\sqrt{2\\log N}$; KL like $\\log N$", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    ax.plot(kl, expected_max, marker="o", color=colors[2])
    for n, k, e in zip(ns, kl, expected_max):
        if n in (1, 4, 16, 64, 256, 1024):
            ax.annotate(f"N={n}", (k, e), textcoords="offset points", xytext=(4, -9), fontsize=7.5)
    ax.set_xlabel("KL(BoN || $\\pi$) in nats")
    ax.set_ylabel("expected proxy reward")
    ax.set_title("Reward per nat of KL spent (Gao et al. axis)", fontsize=10, loc="left")
    for a in axes:
        a.grid(color="#eeeeee", linewidth=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
