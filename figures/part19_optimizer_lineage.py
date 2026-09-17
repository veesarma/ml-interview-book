"""The optimizer lineage: each step fixes one named failure of the step before it.

Two branches leave plain SGD. The upper branch accumulates a direction
(momentum); the lower branch accumulates a per-parameter scale (AdaGrad,
RMSProp). Adam merges the two branches, and AdamW repairs how Adam applies
weight decay.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#9467bd", "#7f7f7f"

W, H = 5.0, 2.8

# name, update rule, what it leaves broken / what it fixes, x, y, colour
NODES = [
    ("SGD", r"$\theta \leftarrow \theta - \eta g$",
     "one global step size,\nzig-zags along ravines", 3.0, 8.6, GREY),
    ("Momentum", r"$v \leftarrow \beta v + g$",
     "damps the oscillation across\nthe ravine, accelerates along it", 9.6, 8.6, BLUE),
    ("AdaGrad", r"$\eta\,/\,\sqrt{\textstyle\sum g^2}$",
     "per-parameter steps, larger for\nrare features; the sum only grows,\nso the step decays to zero", 3.0, 2.2, ORANGE),
    ("RMSProp", r"$\eta\,/\,\sqrt{\mathrm{EMA}(g^2)}$",
     "an EMA instead of a running sum,\nso the denominator stops growing", 9.6, 2.2, ORANGE),
    ("Adam", r"$\mathrm{EMA}(g)\,/\,\sqrt{\mathrm{EMA}(g^2)}$",
     "both moments, with bias correction\nfor the first few steps", 16.2, 5.4, PURPLE),
    ("AdamW", r"decoupled  $\theta \leftarrow \theta - \eta\lambda\theta$",
     "L2 added to the loss gets divided by\nthe adaptive denominator; decay it\nseparately and generalisation improves",
     22.8, 5.4, PURPLE),
]

EDGES = [
    (0, 1, "add a direction"),
    (0, 2, "add a per-parameter scale"),
    (2, 3, "fix the stalling"),
    (1, 4, "first moment"),
    (3, 4, "second moment"),
    (4, 5, "fix the decay"),
]


def endpoints(a, b):
    """Edge exit and entry points for boxes a=(x,y) and b=(x,y)."""
    (xa, ya), (xb, yb) = a, b
    if abs(ya - yb) < 0.1:
        return (xa + W / 2 + 0.1, ya), (xb - W / 2 - 0.1, yb)
    if xb > xa + W / 2:  # diagonal to the right
        sign = 1.0 if yb > ya else -1.0
        return (xa + W / 2 + 0.1, ya + sign * 0.6), (xb - W / 2 - 0.1, yb - sign * 0.6)
    sign = 1.0 if yb > ya else -1.0  # straight up or down
    return (xa, ya + sign * (H / 2 + 0.1)), (xb, yb - sign * (H / 2 + 0.1))


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    ax.set_xlim(0.0, 25.8)
    ax.set_ylim(-0.4, 11.4)
    ax.axis("off")

    for name, rule, note, x, y, color in NODES:
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.16,rounding_size=0.25",
                                    facecolor="white", edgecolor=color, linewidth=1.4))
        ax.text(x, y + 0.95, name, ha="center", va="center", fontsize=11.5, fontweight="bold", color=color)
        ax.text(x, y + 0.32, rule, ha="center", va="center", fontsize=9)
        ax.text(x, y - 0.65, note, ha="center", va="center", fontsize=8.2, color="#333333", linespacing=1.35)

    for a, b, label in EDGES:
        p, q = endpoints((NODES[a][3], NODES[a][4]), (NODES[b][3], NODES[b][4]))
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color="#555555",
                                     linewidth=1.1, shrinkA=1, shrinkB=1))
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.30, label, ha="center", va="center",
                fontsize=8.5, color="#555555", bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

    ax.text(12.9, 10.9, "Each arrow repairs the failure named in the box behind it",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(12.9, -0.15,
            "Defaults worth defending: AdamW for transformers; SGD with momentum stays competitive for large-scale "
            "vision and carries two fewer optimizer states per parameter.",
            ha="center", fontsize=9, style="italic", color="#444444")

    fig.tight_layout()
    path = OUT / "part19_optimizer_lineage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    main()
