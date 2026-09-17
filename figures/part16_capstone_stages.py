"""Accuracy and wall time of every stage of canon #60, measured by running it.

The numbers come from `mlbook.capstone.pipeline.run_pipeline()` at its default
configuration, so this figure cannot drift from the code.

Writes docs/assets/figures/part16_capstone_stages.png.
Run from the repository root: python figures/part16_capstone_stages.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlbook.capstone.pipeline import run_pipeline

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part16_capstone_stages.png"

STAGES = [
    ("majority\nbaseline", "majority_baseline", "#bbbbbb"),
    ("random\ninit", "random_init", "#bbbbbb"),
    ("SFT", "sft", "#1f77b4"),
    ("+ DPO", "dpo", "#ff7f0e"),
    ("+ GRPO", "grpo", "#2ca02c"),
    ("+ tool\nloop", "tool_loop", "#9467bd"),
]
TIMED = [("data", "data"), ("SFT", "sft"), ("reward\nmodel", "reward_model"),
         ("DPO", "dpo"), ("GRPO", "grpo"), ("tool\nloop", "tool_loop")]


def main() -> None:
    report = run_pipeline()
    acc = report["accuracy"]
    times = report["wall_time_s"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150,
                                      gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.patch.set_facecolor("white")

    labels = [name for name, _, _ in STAGES]
    values = [acc[key] for _, key, _ in STAGES]
    colours = [colour for _, _, colour in STAGES]
    bars = left.bar(labels, values, color=colours, width=0.62)
    for bar, value in zip(bars, values):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
                  ha="center", fontsize=9)
    left.axhline(acc["majority_baseline"], color="#999999", linestyle="--", linewidth=1.0)
    left.set_ylim(0, max(values) * 1.22)
    left.set_ylabel("exact-match accuracy on held-out images")
    left.set_title("Every stage, measured the same way", fontsize=11)
    left.spines[["top", "right"]].set_visible(False)

    tlabels = [name for name, _ in TIMED]
    tvalues = [times[key] for _, key in TIMED]
    right.barh(tlabels[::-1], tvalues[::-1], color="#4c72b0", height=0.6)
    for i, value in enumerate(tvalues[::-1]):
        right.text(value + max(tvalues) * 0.02, i, f"{value:.1f}s", va="center", fontsize=9)
    right.set_xlim(0, max(tvalues) * 1.25)
    right.set_xlabel("wall time, one CPU thread")
    right.set_title(f"{times['total']:.0f}s end to end, {report['params']['total']:,} parameters", fontsize=11)
    right.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print({k: round(v, 3) for k, v in acc.items()})


if __name__ == "__main__":
    main()
