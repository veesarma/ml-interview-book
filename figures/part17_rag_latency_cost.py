"""LLM assistant: latency budget by stage and the quality/cost frontier that model routing exploits."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_rag_latency_cost.png"

STAGES = [("Guardrail + intent classifier", 40), ("Query rewrite (small model)", 150), ("Hybrid retrieval + rerank", 120),
          ("Prompt assembly + cache lookup", 20), ("Time to first token (LLM prefill)", 400), ("Streaming (per token)", 0)]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    left = 0.0
    for i, (name, ms) in enumerate(STAGES[:-1]):
        ax1.barh(0, ms, left=left, color=f"C{i}", edgecolor="white", linewidth=2, height=0.5, label=f"{name} ({ms} ms)")
        left += ms
    ax1.set_yticks([])
    ax1.set_xlabel("milliseconds until the first streamed token (illustrative)")
    ax1.set_title("Pre-generation work must stay well under the prefill time")
    ax1.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.7), ncol=2, frameon=False)

    cost = np.array([0.05, 0.15, 0.6, 3.0, 15.0])  # relative $ per 1k requests
    quality = np.array([0.62, 0.71, 0.80, 0.86, 0.88])
    names = ["7B small", "small + RAG", "mid + RAG", "frontier + RAG", "frontier + RAG + reasoning"]
    ax2.plot(cost, quality, "o-", color="C0", lw=2)
    for c, q, n in zip(cost, quality, names):
        ax2.annotate(n, (c, q), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("relative cost per 1k requests (log)")
    ax2.set_ylabel("task pass rate on the eval set")
    ax2.set_title("Routing: send the easy 70% down the cheap path")
    ax2.set_ylim(0.55, 0.95)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
