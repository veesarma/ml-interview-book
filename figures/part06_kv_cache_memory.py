"""KV-cache memory vs context length for several production models (batch = 1, bf16),
with the 80 GB HBM line of an H100 as a reference.

Writes docs/assets/figures/part06_kv_cache_memory.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.llm.kv_cache_calc import (  # noqa: E402
    LLAMA2_7B,
    LLAMA2_70B,
    LLAMA3_8B,
    LLAMA3_70B,
    LLAMA3_70B_MHA,
    LLAMA3_405B,
    kv_cache_bytes,
    mla_bytes_per_token,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_kv_cache_memory.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    T = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor="white")
    specs = [LLAMA2_7B, LLAMA3_8B, LLAMA2_70B, LLAMA3_70B, LLAMA3_70B_MHA, LLAMA3_405B]
    for i, spec in enumerate(specs):
        gib = np.array([kv_cache_bytes(spec, int(t)) for t in T]) / 1024**3
        style = "--" if "if MHA" in spec.name else "-"
        ax.plot(T, gib, style, marker="o", markersize=3.5, linewidth=1.8, color=colors[i], label=spec.name)
        ax.annotate(spec.name, (T[-1], gib[-1]), textcoords="offset points", xytext=(5, -3), fontsize=7.5, color=colors[i])
    mla = np.array([t * mla_bytes_per_token(60, 512, 64, 2) for t in T]) / 1024**3  # DeepSeek-V2: 60 layers
    ax.plot(T, mla, ":", marker="o", markersize=3.5, linewidth=1.8, color=colors[6], label="DeepSeek-V2 MLA (60 layers, 576/token/layer)")
    ax.annotate("DeepSeek-V2 MLA", (T[-1], mla[-1]), textcoords="offset points", xytext=(5, -3), fontsize=7.5, color=colors[6])
    ax.axhline(80, color="#888888", linewidth=0.8, linestyle="-.")
    ax.text(T[0], 90, "80 GB (one H100, before weights)", fontsize=7.5, color="#555555")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(T[0] * 0.9, T[-1] * 6)
    ax.set_xlabel("context length T (tokens), batch = 1")
    ax.set_ylabel("KV cache (GiB, bf16)")
    ax.set_title("KV-cache memory = 2 · L · T · H_kv · d_head · 2 bytes", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(color="#eeeeee", linewidth=0.5, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
