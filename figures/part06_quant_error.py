"""Round-to-nearest quantisation error vs bit width and group size on a weight matrix
with a heavy-tailed (outlier) column, plus the float-format precision/range table.

Writes docs/assets/figures/part06_quant_error.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mlbook.quant.quantize import BF16, FP8_E4M3, FP8_E5M2, FP16, FP32, quantization_mse  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_quant_error.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    W = torch.randn(256, 1024) * 0.02
    W[:, 17] *= 40.0  # an outlier input channel, typical of LLM weights after training
    group_sizes = [None, 1024, 256, 128, 64, 32]
    labels = ["per-tensor", "per-row (g=1024)", "g=256", "g=128", "g=64", "g=32"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white", gridspec_kw={"width_ratios": [1.3, 1]})
    for i, bits in enumerate([8, 6, 4, 3, 2]):
        errs = [quantization_mse(W, bits, g) / (W**2).mean().item() for g in group_sizes]
        ax1.plot(range(len(group_sizes)), errs, marker="o", linewidth=1.8, color=colors[i], label=f"INT{bits}")
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(group_sizes)))
    ax1.set_xticklabels(labels, fontsize=8, rotation=15)
    ax1.set_ylabel("relative MSE  ‖W − Ŵ‖² / ‖W‖²")
    ax1.set_title("Round-to-nearest error vs bits and group size (outlier column present)", fontsize=9.5, loc="left")
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(color="#eeeeee", linewidth=0.5, which="both")
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    ax2.axis("off")
    rows = [(f.name, f.exp_bits, f.mant_bits, f"{f.max_normal:.3g}", f"{f.epsilon:.2e}") for f in (FP32, FP16, BF16, FP8_E4M3, FP8_E5M2)]
    table = ax2.table(cellText=rows, colLabels=["format", "exp bits", "mant bits", "max", "ε = 2^-m"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#f0f0f0")
    ax2.set_title("Float formats: exponent buys range, mantissa buys precision", fontsize=9.5, loc="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
