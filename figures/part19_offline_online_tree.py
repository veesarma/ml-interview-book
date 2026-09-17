"""The 'offline good, online bad' triage tree, drawn as the first cut you run live.

One experiment (log the production feature vectors, score them offline with the
deployed artifact) splits the hypothesis space in half before you touch the
model. Each leaf names the confirming test, not just the hypothesis.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"


def box(ax, x, y, w, h, text, face, edge, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.18,rounding_size=0.25",
                                facecolor=face, edgecolor=edge, linewidth=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, fontweight=weight, color="black", linespacing=1.35)


def arrow(ax, p, q, label=None, color="black", dx=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=1.1, shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2, label, ha="center", va="center",
                fontsize=8.5, color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13)
    ax.axis("off")

    box(ax, 10, 12.2, 8.6, 1.0, "Offline metric good, online metric bad", "#f2f2f2", GREY, 11, "bold")
    box(ax, 10, 10.4, 9.4, 1.2,
        "Log-and-replay: capture the feature vectors actually served,\n"
        "score them offline with the deployed artifact", "#fff4e6", ORANGE, 9.5, "bold")
    arrow(ax, (10, 11.6), (10, 11.05))

    box(ax, 4.6, 8.4, 7.2, 1.0, "Replay reproduces the BAD number\n(the data is the problem)", "#eaf3fb", BLUE, 9.5)
    box(ax, 15.4, 8.4, 7.2, 1.0, "Replay reproduces the GOOD number\n(serving or measurement is the problem)",
        "#eaf7ec", GREEN, 9.5)
    arrow(ax, (9.2, 9.8), (5.6, 9.0))
    arrow(ax, (10.8, 9.8), (14.4, 9.0))

    left = [
        ("Training/serving skew", "diff the offline and online feature code paths;\ncompare per-feature histograms and null rates"),
        ("Covariate or concept shift", "PSI / KS per feature train vs live;\nscore the freshest labelled slice"),
        ("Leakage in the offline set", "audit each feature's as-of time;\ndrop the suspect and re-measure"),
    ]
    right = [
        ("The served artifact is not the eval artifact", "re-run the offline eval on the exported,\nquantised, distilled binary"),
        ("Metric proxy mismatch", "check the historical correlation between\nthe offline metric and the KPI"),
        ("Feedback loop / selection bias", "labels exist only for what you served;\nlook for a logging policy and its propensities"),
    ]
    for column, items, x, color in [("left", left, 4.6, BLUE), ("right", right, 15.4, GREEN)]:
        parent_bottom = 8.4 - 0.68
        for i, (title, test) in enumerate(items):
            y = 6.4 - i * 2.05
            box(ax, x, y, 7.6, 1.5, f"{title}\n{test}", "white", color, 8.7)
            arrow(ax, (x, parent_bottom), (x, y + 0.93), color=color)
            parent_bottom = y - 0.93

    ax.text(10, 0.35, "Run the cheapest discriminating test first. Do not change the model until you know which half you are in.",
            ha="center", fontsize=9.5, style="italic", color="#444444")

    fig.tight_layout()
    path = OUT / "part19_offline_online_tree.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    main()
