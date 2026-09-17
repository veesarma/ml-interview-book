"""AV perception: an onboard latency budget (illustrative) and the long-tail data-engine curve."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_av_latency_budget.png"

STAGES = [("Sensor capture + ISP", 8), ("Backbone (8 cams)", 22), ("BEV/occupancy head", 12),
          ("Detection + tracking", 8), ("Prediction", 10), ("Planning", 15), ("Control + actuation", 5)]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1.0, 1.6]})
    left = 0.0
    for i, (name, ms) in enumerate(STAGES):
        ax1.barh(0, ms, left=left, color=f"C{i}", edgecolor="white", linewidth=2, height=0.5, label=f"{name} ({ms} ms)")
        left += ms
    ax1.axvline(100, color="k", ls="--", lw=1)
    ax1.text(99, -0.42, "100 ms end-to-end target (10 Hz loop)", fontsize=8, ha="right")
    ax1.set_yticks([])
    ax1.set_xlim(0, 110)
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_xlabel("milliseconds (illustrative budget; interviewer sets the real one)")
    ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    ax1.set_title("Every stage borrows from the same 100 ms", pad=28)

    rng = np.random.default_rng(0)
    freq = np.sort(rng.pareto(1.2, 400) + 1)[::-1]
    freq = freq / freq.sum()
    ax2.plot(np.arange(1, 401), freq, color="C0", lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("scenario type (ranked by frequency)")
    ax2.set_ylabel("share of fleet miles (log)")
    ax2.set_title("The long tail: the data engine spends its budget on the right side")
    ax2.axvspan(40, 400, color="C1", alpha=0.12)
    ax2.text(200, freq[5], "trigger-mined, active-learned,\nsimulated, over-sampled", fontsize=8, ha="center")
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
