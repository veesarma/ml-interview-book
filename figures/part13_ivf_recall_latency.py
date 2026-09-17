"""Recall@10 and query time versus nprobe for an IVF index (synthetic data).

Writes docs/assets/figures/part13_ivf_recall_latency.png. Run from the repo root:
    python figures/part13_ivf_recall_latency.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.retrieval.ivf import IVFIndex  # noqa: E402
from mlbook.retrieval.similarity import brute_force_topk  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_ivf_recall_latency.png"


def main() -> None:
    rng = np.random.default_rng(0)
    N, d, n_list, n_q = 10000, 32, 100, 40
    centers = rng.normal(size=(30, d)) * 1.0  # mild cluster structure, like real embeddings
    X = centers[rng.integers(0, 30, N)] + rng.normal(size=(N, d))  # (N, d) corpus
    Q = centers[rng.integers(0, 30, n_q)] + rng.normal(size=(n_q, d))  # (n_q, d)
    truth, _ = brute_force_topk(Q, X, 10, "l2")
    idx = IVFIndex(n_list=n_list, n_iters=10).train(X).add(X)
    t0 = time.perf_counter()
    for i in range(n_q):
        brute_force_topk(Q[i : i + 1], X, 10, "l2")
    t_brute = (time.perf_counter() - t0) / n_q * 1e3
    probes = [1, 2, 4, 8, 16, 32, 64, 100]
    recall, latency = [], []
    for nprobe in probes:
        hits, t0 = 0, time.perf_counter()
        for i in range(n_q):
            ids, _ = idx.search(Q[i], 10, nprobe=nprobe)
            hits += len(set(ids) & set(truth[i]))
        latency.append((time.perf_counter() - t0) / n_q * 1e3)
        recall.append(hits / (10 * n_q))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(probes, recall, marker="o", label="recall@10")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("nprobe (of 100 lists)")
    ax1.set_ylabel("recall@10 vs exact search")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(probes, latency, marker="s", color="C1", label="query time (ms)")
    ax2.axhline(t_brute, color="C3", ls="--", label=f"brute force ({t_brute:.1f} ms)")
    ax2.set_ylabel("ms per query (NumPy, single thread)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title(f"IVF-Flat on N={N}, d={d}: recall rises with nprobe, so does cost")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
