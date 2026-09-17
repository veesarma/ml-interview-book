"""FixMatch: the weak/strong pipeline, the confidence threshold, and confirmation bias.

Writes docs/assets/figures/part10_fixmatch.png.  Run:  python figures/part10_fixmatch.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from mlbook.ssl import fixmatch as FX  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part10_fixmatch.png"
torch.set_num_threads(1)
THRESHOLDS = (0.0, 0.5, 0.8, 0.95, 0.99)


def make_data(n_lab: int = 4, imbalance: float = 0.5):
    """Two clusters. ``imbalance`` is the fraction of unlabelled points from class 1."""
    centers = torch.tensor([[-1.2, 0.0], [1.2, 0.0]])
    y_u = (torch.rand(400) < imbalance).long()
    x_u = centers[y_u] + 1.3 * torch.randn(400, 2)     # the clusters overlap, so the threshold matters
    y_l = torch.arange(2).repeat(n_lab // 2)
    x_l = centers[y_l] + 0.3 * torch.randn(n_lab, 2)
    return x_l, y_l, x_u, y_u


def train(threshold: float, imbalance: float = 0.5, steps: int = 400):
    torch.manual_seed(0)
    x_l, y_l, x_u, y_u = make_data(imbalance=imbalance)
    net = torch.nn.Sequential(torch.nn.Linear(2, 48), torch.nn.ReLU(), torch.nn.Linear(48, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        loss, frac = FX.fixmatch_loss(net(x_l), y_l, net(FX.weak_augment(x_u, 0.1)),
                                      net(FX.strong_augment(x_u, 0.5, 0.0)),
                                      threshold=threshold, lambda_u=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = net(x_u)
        acc = (logits.argmax(1) == y_u).float().mean().item()
        conf, pseudo = F.softmax(logits, 1).max(1)
        kept = conf >= max(threshold, 1e-6)
        prec = (pseudo[kept] == y_u[kept]).float().mean().item() if kept.any() else float("nan")
        share1 = pseudo[kept].float().mean().item() if kept.any() else float("nan")
    return acc, float(kept.float().mean()), prec, share1


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure(figsize=(14, 6.4), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.0], hspace=0.42, wspace=0.28)

    # top: the pipeline
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    nodes = [
        (0.01, 0.35, 0.13, 0.42, "unlabelled\nimage $x_u$", 7),
        (0.19, 0.60, 0.15, 0.34, "weak aug\n(flip, crop)", 0),
        (0.19, 0.06, 0.15, 0.34, "strong aug\n(RandAugment,\nCutout)", 1),
        (0.39, 0.60, 0.13, 0.34, "model\n$p_\\theta$", 2),
        (0.39, 0.06, 0.13, 0.34, "model\n$p_\\theta$", 2),
        (0.57, 0.60, 0.17, 0.34, "argmax if\n$\\max p \\geq \\tau$\n(stop-gradient)", 3),
        (0.57, 0.06, 0.17, 0.34, "prediction\n(gradient flows)", 3),
        (0.80, 0.33, 0.18, 0.40, "cross-entropy\non the accepted\nexamples only", 4),
    ]
    for x, y, w, h, label, ci in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", linewidth=1.3,
                                    edgecolor=colors[ci], facecolor=colors[ci], alpha=0.12))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, color="#222222")
    arrows = [(0.14, 0.60, 0.19, 0.77), (0.14, 0.50, 0.19, 0.23), (0.34, 0.77, 0.39, 0.77),
              (0.34, 0.23, 0.39, 0.23), (0.52, 0.77, 0.57, 0.77), (0.52, 0.23, 0.57, 0.23),
              (0.74, 0.77, 0.80, 0.60), (0.74, 0.23, 0.80, 0.45)]
    for x0, y0, x1, y1 in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color="#777777"))
    ax.text(0.5, 1.0, "FixMatch: the weak view writes the label, the strong view takes the gradient",
            ha="center", va="top", fontsize=12, fontweight="bold")

    # bottom left: accuracy and coverage against the threshold
    accs, fracs, precs = [], [], []
    for t in THRESHOLDS:
        a, f, p, _ = train(t)
        accs.append(a); fracs.append(f); precs.append(p)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(THRESHOLDS, accs, "o-", color=colors[0], label="accuracy on the unlabelled pool")
    ax.plot(THRESHOLDS, precs, "s-", color=colors[2], label="precision of accepted pseudo-labels")
    ax.plot(THRESHOLDS, fracs, "^--", color=colors[1], label="fraction accepted (coverage)")
    ax.set_xlabel(r"confidence threshold $\tau$"); ax.set_ylim(0, 1.05)
    ax.set_title("Coverage falls, pseudo-label precision rises", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    # bottom middle: what happens under class imbalance
    ax = fig.add_subplot(gs[1, 1])
    imbalances = [0.5, 0.7, 0.8, 0.9]
    shares, accs2 = [], []
    for im in imbalances:
        a, _, _, s1 = train(0.95, imbalance=im)
        shares.append(s1); accs2.append(a)
    ax.plot(imbalances, shares, "o-", color=colors[3], label="share of accepted labels that are class 1")
    ax.plot(imbalances, imbalances, "--", color="#888888", label="true class-1 share")
    ax.set_xlabel("true class-1 share of the unlabelled pool")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accepted pseudo-labels over-represent the majority", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # bottom right: the decision boundary with and without the unlabelled term
    ax = fig.add_subplot(gs[1, 2])
    torch.manual_seed(0)
    x_l, y_l, x_u, y_u = make_data()
    ax.scatter(x_u[:, 0], x_u[:, 1], s=6, alpha=0.3, c=[colors[int(v)] for v in y_u])
    ax.scatter(x_l[:, 0], x_l[:, 1], s=110, marker="*", edgecolor="black",
               c=[colors[int(v)] for v in y_l], zorder=3, label="the 4 labels")
    for use_u, style in ((False, "--"), (True, "-")):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(2, 48), torch.nn.ReLU(), torch.nn.Linear(48, 2))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(400):
            if use_u:
                loss, _ = FX.fixmatch_loss(net(x_l), y_l, net(FX.weak_augment(x_u, 0.1)),
                                           net(FX.strong_augment(x_u, 0.5, 0.0)), 0.95, 1.0)
            else:
                loss = F.cross_entropy(net(x_l), y_l)
            opt.zero_grad(); loss.backward(); opt.step()
        g = torch.linspace(-4.5, 4.5, 140)
        gx, gy = torch.meshgrid(g, g, indexing="xy")
        with torch.no_grad():
            p = F.softmax(net(torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)), 1)[:, 1]
        ax.contour(gx, gy, p.reshape(gx.shape), levels=[0.5], colors="black", linestyles=style, linewidths=1.6)
    ax.set_title("solid: with the FixMatch term\ndashed: the 4 labels alone", fontsize=10)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-3.2, 3.2)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    for ax in fig.axes[1:]:
        ax.grid(alpha=0.22, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
