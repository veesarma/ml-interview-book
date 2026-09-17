"""Uplift modelling: the four quadrants, and CUPED variance reduction on a simulated A/B test."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_uplift_quadrants.png"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="k", lw=1)
    ax1.axvline(0.5, color="k", lw=1)
    labels = {
        (0.25, 0.75): ("Persuadables\nP(y|T)=high, P(y|C)=low\n-> SEND", "C2"),
        (0.75, 0.75): ("Sure things\nconvert either way\n-> save the budget", "C0"),
        (0.25, 0.25): ("Lost causes\nnever convert\n-> skip", "C7"),
        (0.75, 0.25): ("Sleeping dogs\nsending HURTS\n-> never send", "C3"),
    }
    for (x, y), (t, c) in labels.items():
        ax1.add_patch(plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, color=c, alpha=0.15))
        ax1.text(x, y, t, ha="center", va="center", fontsize=9)
    ax1.set_xlabel("P(convert | no notification)")
    ax1.set_ylabel("P(convert | notification)")
    ax1.set_title("Target uplift = P(y|T) - P(y|C), not P(y|T)")
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_yticks([0, 0.5, 1])

    rng = np.random.default_rng(2)
    n = 4000
    pre = rng.normal(50, 15, n)                      # pre-experiment metric (covariate)
    treat = rng.random(n) < 0.5
    post = 0.8 * pre + rng.normal(0, 8, n) + 1.5 * treat   # true effect 1.5
    theta = np.cov(post, pre)[0, 1] / np.var(pre)
    post_cuped = post - theta * (pre - pre.mean())
    for data, c, name in [(post, "C0", "raw metric"), (post_cuped, "C2", "CUPED-adjusted")]:
        diff = data[treat].mean() - data[~treat].mean()
        se = np.sqrt(data[treat].var() / treat.sum() + data[~treat].var() / (~treat).sum())
        ax2.errorbar([name], [diff], yerr=[1.96 * se], fmt="o", color=c, capsize=6, ms=8, lw=2)
    ax2.axhline(1.5, color="k", ls="--", lw=1, label="true effect = 1.5")
    ax2.axhline(0, color="0.6", lw=1)
    ax2.set_ylabel("estimated treatment effect (95% CI)")
    ax2.set_title(f"CUPED shrinks the CI (variance x {np.var(post_cuped) / np.var(post):.2f})")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
