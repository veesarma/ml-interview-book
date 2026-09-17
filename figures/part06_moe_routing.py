"""Diagram of a top-2 MoE layer: tokens -> router -> two of E experts -> weighted sum,
with a bar chart of expert load before/after the balance loss.

Writes docs/assets/figures/part06_moe_routing.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from mlbook.llm.moe import TopKRouter, expert_usage_fraction, load_balancing_loss  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_moe_routing.png"


def _box(ax, x, y, w, h, text, color, fontsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="none", alpha=0.85))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="black")


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white", gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: routing diagram for one token with E=4 experts, k=2.
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    _box(ax1, 0.3, 2.5, 1.6, 1.0, "token x\n(d_model)", "#dddddd")
    _box(ax1, 2.6, 2.5, 1.8, 1.0, "router\nsoftmax(x W_g)", colors[0])
    probs = [0.55, 0.05, 0.32, 0.08]
    for i, p in enumerate(probs):
        y = 4.9 - i * 1.35
        chosen = i in (0, 2)
        _box(ax1, 5.4, y - 0.4, 1.8, 0.8, f"expert E{i}", colors[2] if chosen else "#e6e6e6")
        arrow = FancyArrowPatch((4.4, 3.0), (5.4, y), arrowstyle="->", mutation_scale=10,
                                linewidth=2.0 if chosen else 0.8, color="black" if chosen else "#bbbbbb",
                                linestyle="-" if chosen else "--")
        ax1.add_patch(arrow)
        ax1.text(4.75, (3.0 + y) / 2 + (0.22 if y > 3 else -0.3), f"p={p:.2f}", fontsize=7, ha="center",
                 color="black" if chosen else "#999999")
        if chosen:
            g = p / (probs[0] + probs[2])
            ax1.add_patch(FancyArrowPatch((7.2, y), (8.3, 3.0), arrowstyle="->", mutation_scale=10, linewidth=2.0))
            ax1.text(7.85, (y + 3.0) / 2 + (0.25 if y > 3 else -0.35), f"g={g:.2f}", fontsize=7, ha="center")
    _box(ax1, 8.3, 2.5, 1.5, 1.0, "Σ g_i E_i(x)", colors[1])
    ax1.add_patch(FancyArrowPatch((1.9, 3.0), (2.6, 3.0), arrowstyle="->", mutation_scale=10, linewidth=1.5))
    ax1.text(5.0, 0.15, "top-2 of 4 experts: dashed experts do no work for this token; g renormalises the two kept probabilities",
             fontsize=7.5, ha="center", color="#444444")
    ax1.set_title("Top-k routing in one MoE layer", fontsize=10, loc="left")

    # Right: expert load before/after optimising the balance loss (same experiment as the test).
    torch.manual_seed(1)
    E = 8
    router = TopKRouter(d_model=16, n_experts=E, k=2)
    with torch.no_grad():
        router.gate.weight[0] += 3.0
    x = torch.randn(512, 16)
    _, idx0, _ = router(x)
    before = expert_usage_fraction(idx0, E).numpy()
    opt = torch.optim.SGD(router.parameters(), lr=1.0)
    for _ in range(300):
        _, idx, probs = router(x)
        loss = load_balancing_loss(probs, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, idx1, _ = router(x)
    after = expert_usage_fraction(idx1, E).numpy()
    xs = range(E)
    ax2.bar([i - 0.2 for i in xs], before, width=0.4, color=colors[3], label="before balance loss")
    ax2.bar([i + 0.2 for i in xs], after, width=0.4, color=colors[2], label="after 300 steps of $L_{aux}$")
    ax2.axhline(1 / E, color="#888888", linewidth=0.8, linestyle="--")
    ax2.text(E - 0.5, 1 / E + 0.01, "1/E", fontsize=7.5, ha="right", color="#555555")
    ax2.set_xticks(list(xs))
    ax2.set_xlabel("expert")
    ax2.set_ylabel("fraction of routed tokens")
    ax2.set_title("Load balancing: expert usage", fontsize=10, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.grid(axis="y", color="#eeeeee", linewidth=0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
