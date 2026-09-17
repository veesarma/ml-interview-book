"""Static vs continuous batching: slot occupancy over time for 8 requests, 4 slots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlbook.systems.continuous_batching_sim import CostModel, Request, simulate_continuous, simulate_static

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part14_batching_timeline.png"


def main() -> None:
    reqs = [Request(0, 0.0, 64, 20), Request(1, 0.0, 64, 4), Request(2, 0.0, 64, 12), Request(3, 0.0, 64, 6),
            Request(4, 0.05, 64, 10), Request(5, 0.05, 64, 3), Request(6, 0.1, 64, 16), Request(7, 0.1, 64, 5)]
    cost = CostModel(t_fixed=0.004, c_prefill=1e-5, c_decode=1e-3)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, (name, sim) in zip(axes, [("static batching", simulate_static), ("continuous batching", simulate_continuous)]):
        m = sim(reqs, 4, cost)
        # assign slots greedily in completion order for drawing
        slots_free = [0.0] * 4
        for c in sorted(m.completed, key=lambda c: c.first_token):
            start = c.first_token
            slot = min(range(4), key=lambda i: (slots_free[i] > start + 1e-9, slots_free[i]))
            ax.barh(slot, c.finish - c.arrival, left=c.arrival, height=0.7, color=plt.cm.tab10(c.rid), alpha=0.85)
            ax.text(c.arrival + (c.finish - c.arrival) / 2, slot, f"r{c.rid} ({c.gen_len} tok)", ha="center", va="center", fontsize=7)
            slots_free[slot] = c.finish
        ax.set_yticks(range(4)); ax.set_yticklabels([f"slot {i}" for i in range(4)])
        ax.set_title(f"{name}: makespan {m.makespan * 1e3:.0f} ms, throughput {m.throughput:.0f} tok/s, mean latency {m.mean_latency * 1e3:.0f} ms", fontsize=9)
    axes[1].set_xlabel("time (s); each bar spans arrival -> last token (static: slots idle while the longest request in the batch finishes)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
