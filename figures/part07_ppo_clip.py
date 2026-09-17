"""The PPO clipped objective min(r A, clip(r, 1-eps, 1+eps) A) as a function of the ratio r,
for positive and negative advantage. Writes docs/assets/figures/part07_ppo_clip.png.
Run from the repository root:  python figures/part07_ppo_clip.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part07_ppo_clip.png"


def clipped(r: np.ndarray, A: float, eps: float) -> np.ndarray:
    return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)


def main() -> None:
    eps = 0.2
    r = np.linspace(0.4, 1.6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor="white")
    for ax, A, title in ((axes[0], 1.0, "advantage $A_t > 0$"), (axes[1], -1.0, "advantage $A_t < 0$")):
        ax.plot(r, r * A, linestyle="--", color="#999999", linewidth=1.2, label="unclipped $r_t A_t$")
        ax.plot(r, clipped(r, A, eps), linewidth=2.2, label="$L^{CLIP}$ term")
        ax.axvline(1 - eps, color="#cccccc", linewidth=0.8)
        ax.axvline(1 + eps, color="#cccccc", linewidth=0.8)
        ax.axvline(1.0, color="#cccccc", linewidth=0.8, linestyle=":")
        flat_from = 1 + eps if A > 0 else 1 - eps
        ax.annotate("zero gradient:\nno incentive to move\nthe ratio further", xy=(flat_from + (0.2 if A > 0 else -0.2), clipped(np.array([flat_from]), A, eps)[0]),
                    xytext=(flat_from + (0.05 if A > 0 else -0.55), (0.6 if A > 0 else -1.5)), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8))
        ax.set_xlabel("probability ratio  $r_t(\\theta) = \\pi_\\theta(a_t|s_t) / \\pi_{old}(a_t|s_t)$")
        ax.set_title(title + f"  ($\\epsilon$ = {eps})", fontsize=10, loc="left")
        ax.grid(color="#eeeeee", linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left" if A > 0 else "lower left")
    axes[0].set_ylabel("objective (maximise)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
