"""Visual search: recall@10 vs queries-per-second for ANN index families (illustrative shapes)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_ann_tradeoff.png"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # Left: recall vs QPS frontier per index family (illustrative, shaped like ann-benchmarks plots).
    qps = np.logspace(2, 5, 50)
    families = {
        "flat / exact (brute force)": (0.999, 0.0, 300),
        "HNSW (graph, fp32)": (0.99, 0.25, 8_000),
        "IVF-PQ (compressed)": (0.93, 0.35, 30_000),
    }
    for (name, (top, slope, knee)), c in zip(families.items(), ["C0", "C1", "C2"]):
        recall = top - slope * np.clip(np.log10(qps / knee), 0, None)
        recall = np.where(qps > knee * 4, np.nan, recall)
        ax1.plot(qps, np.clip(recall, 0, 1), lw=2, color=c, label=name)
    ax1.set_xscale("log")
    ax1.set_xlabel("queries per second per node (log)")
    ax1.set_ylabel("recall@10 vs exact search")
    ax1.set_ylim(0.6, 1.01)
    ax1.set_title("Recall-throughput frontier (illustrative)")
    ax1.legend(fontsize=8, loc="lower left")

    # Right: memory per vector by representation, for a 1B x 128-d index.
    reps = ["fp32\n(512 B)", "fp16\n(256 B)", "int8 SQ\n(128 B)", "PQ 32B", "PQ 16B"]
    bytes_per = np.array([512, 256, 128, 32, 16])
    tb = bytes_per * 1e9 / 1e12
    ax2.bar(reps, tb, color="C0", width=0.6)
    for i, t in enumerate(tb):
        ax2.text(i, t + 0.01, f"{t:.3f} TB", ha="center", fontsize=8)
    ax2.set_ylabel("index memory for 1B vectors (TB)")
    ax2.set_title("Compression decides whether the index fits in RAM")
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
