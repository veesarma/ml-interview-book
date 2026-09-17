"""Arithmetic intensity of prefill and decode vs the H100 ridge point, and the
resulting per-step time as a function of batch size (Llama-2-7B, bf16)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.systems.inference_metrics import H100_SXM, decode_ridge_batch, estimate_latency
from mlbook.systems.memory_calc import LLAMA2_7B

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part14_prefill_decode_intensity.png"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    B = np.logspace(0, 3, 60)
    ax = axes[0]
    ax.loglog(B, B, label="decode step: intensity ~ B (FLOP/byte, bf16)", color="C0", lw=2)
    ax.axhline(H100_SXM.ridge_point, color="k", ls="--", lw=1, label=f"H100 ridge ~ {H100_SXM.ridge_point:.0f} FLOP/byte")
    ax.axhline(512, color="C1", ls=":", lw=2, label="prefill of a 512-token prompt: intensity ~ 512")
    ax.axvline(decode_ridge_batch(H100_SXM), color="0.5", lw=1)
    ax.text(decode_ridge_batch(H100_SXM) * 1.05, 2, f"B* ~ {decode_ridge_batch(H100_SXM):.0f}", fontsize=9)
    ax.fill_between(B, 1, np.minimum(B, H100_SXM.ridge_point), alpha=0.08, color="C0")
    ax.set_xlabel("batch size B (sequences per decode step)")
    ax.set_ylabel("arithmetic intensity (FLOP / byte)")
    ax.set_title("Decode is memory-bound until B reaches the ridge", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    Bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    tpot = [estimate_latency(LLAMA2_7B, H100_SXM, int(b), 512, 256, efficiency=1.0).tpot_s * 1e3 for b in Bs]
    tps = [estimate_latency(LLAMA2_7B, H100_SXM, int(b), 512, 256, efficiency=1.0).tokens_per_s for b in Bs]
    ax.semilogx(Bs, tpot, "o-", color="C0", label="TPOT (ms per token per sequence)")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("TPOT (ms)", color="C0")
    ax2 = ax.twinx()
    ax2.semilogx(Bs, tps, "s--", color="C3", label="throughput (tokens/s)")
    ax2.set_ylabel("tokens / s", color="C3")
    ax.set_title("Llama-2-7B on one H100 (roofline, datasheet peaks): batching is nearly free until B*", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
