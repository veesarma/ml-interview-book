"""GPipe vs 1F1B schedules for p=4 stages, m=8 micro-batches (t_b = 2 t_f)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlbook.systems.pipeline_calc import bubble_fraction, peak_in_flight, simulate_schedule

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part14_pipeline_bubble.png"


def main() -> None:
    p, m = 4, 8
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    for ax, sch in zip(axes, ["gpipe", "1f1b"]):
        ops = simulate_schedule(sch, p, m, 1.0, 2.0)
        for o in ops:
            color = plt.cm.tab10(o.micro % 10)
            ax.barh(p - 1 - o.stage, o.end - o.start, left=o.start, height=0.8, color=color,
                    alpha=0.95 if o.kind == "F" else 0.45, edgecolor="white", lw=0.6)
            ax.text(o.start + (o.end - o.start) / 2, p - 1 - o.stage, f"{o.kind}{o.micro}", ha="center", va="center", fontsize=6.5)
        ax.set_yticks(range(p))
        ax.set_yticklabels([f"stage {p - 1 - i}" for i in range(p)])
        ax.set_title(f"{sch.upper()}: bubble fraction (p-1)/(m+p-1) = {bubble_fraction(p, m):.3f}, "
                     f"peak micro-batches in flight per stage = {peak_in_flight(ops, p)}", fontsize=9)
    axes[1].set_xlabel("time (units of t_f; solid = forward, faded = backward, colour = micro-batch)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
