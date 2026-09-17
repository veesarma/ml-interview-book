"""Multimodal trajectory prediction: why averaging fails, and what minADE measures.

Writes docs/assets/figures/part11_trajectories.png
Run from the repository root:  python figures/part11_trajectories.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_trajectories.png"

T = 12
S = np.linspace(0.0, 1.0, T)


def arc(curvature: float, length: float = 30.0) -> np.ndarray:
    """A constant-curvature path of T points starting at the origin heading +x."""
    x = length * S
    y = curvature * (length * S) ** 2 / 2.0 / 30.0
    return np.stack([x, y], axis=1)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), facecolor="white")

    left, straight, right = arc(0.55), arc(0.0), arc(-0.55)
    hist = np.stack([np.linspace(-12, 0, 6), np.zeros(6)], axis=1)

    # ---- 1. the three real futures, and the mean of them -----------------------------
    ax = axes[0]
    for traj, lab, c in [(left, "turn left", colors[0]), (straight, "go straight", colors[2]),
                         (right, "turn right", colors[1])]:
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3, color=c, lw=2.0, label=lab)
    mean = (left + straight + right) / 3.0
    ax.plot(mean[:, 0], mean[:, 1], "--", color="#b00020", lw=2.2, label="mean of the three")
    ax.plot(hist[:, 0], hist[:, 1], color="#666666", lw=2.0)
    ax.scatter([0], [0], marker="s", s=55, color="black", zorder=5)
    ax.add_patch(plt.Rectangle((14, -3.0), 8, 6.0, facecolor="#b00020", alpha=0.12))
    ax.text(18, 3.6, "the mean drives\ninto the median", fontsize=8, ha="center", color="#b00020")
    ax.set_title("regressing the mean of a multimodal future", fontsize=9.8, loc="left")
    ax.set_xlabel("x (m)", fontsize=9); ax.set_ylabel("y (m)", fontsize=9)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- 2. six modes with probabilities, the winner highlighted ---------------------
    ax = axes[1]
    rng = np.random.default_rng(1)
    modes = [arc(k) + rng.normal(0, 0.25, size=(T, 2)) for k in (0.75, 0.5, 0.05, -0.05, -0.5, -0.8)]
    probs = np.array([0.08, 0.27, 0.19, 0.16, 0.24, 0.06])
    gt = left
    ades = [np.linalg.norm(m - gt, axis=1).mean() for m in modes]
    best = int(np.argmin(ades))
    for i, m in enumerate(modes):
        ax.plot(m[:, 0], m[:, 1], color=colors[0] if i == best else "#bbbbbb",
                lw=1.0 + 6.0 * probs[i], alpha=1.0 if i == best else 0.75, zorder=3 if i == best else 2)
        ax.text(m[-1, 0] + 0.8, m[-1, 1], f"{probs[i]:.2f}", fontsize=7.5,
                color=colors[0] if i == best else "#888888", va="center")
    ax.plot(gt[:, 0], gt[:, 1], "--", color="black", lw=2.0, label="ground truth")
    ax.plot(hist[:, 0], hist[:, 1], color="#666666", lw=2.0)
    for k in range(0, T, 3):
        ax.plot([gt[k, 0], modes[best][k, 0]], [gt[k, 1], modes[best][k, 1]], color=colors[3], lw=0.9)
    ax.scatter([0], [0], marker="s", s=55, color="black", zorder=5)
    ax.set_title(f"6 modes; the winner (minADE = {ades[best]:.2f} m) takes the gradient", fontsize=9.8, loc="left")
    ax.text(2, -13.5, "line width is the mode probability;\nred segments are the per-step\ndisplacement errors that ADE averages",
            fontsize=8, color="#333333")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- 3. minADE_k as a function of k ----------------------------------------------
    ax = axes[2]
    order = np.argsort(-probs)
    ks = np.arange(1, 7)
    min_ade_k = [min(ades[i] for i in order[:k]) for k in ks]
    min_fde_k = [min(np.linalg.norm(modes[i][-1] - gt[-1]) for i in order[:k]) for k in ks]
    ax.plot(ks, min_ade_k, "-o", color=colors[0], lw=1.9, label="minADE$_k$")
    ax.plot(ks, min_fde_k, "-s", color=colors[1], lw=1.9, label="minFDE$_k$")
    ax.axhline(2.0, color="#b00020", ls="--", lw=1.2)
    ax.text(5.6, 2.25, "miss threshold", fontsize=8, color="#b00020", ha="right")
    ax.set_xlabel("k, modes kept by probability", fontsize=9)
    ax.set_ylabel("metres", fontsize=9)
    ax.set_title("more modes always lowers minADE", fontsize=9.8, loc="left")
    ax.text(1.05, max(min_fde_k) * 0.55, "which is why the metric is\nreported at a fixed k (6 on WOMD,\n6 on Argoverse) and paired with\na probability-aware metric",
            fontsize=8, color="#333333")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
