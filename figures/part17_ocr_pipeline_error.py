"""OCR pipelines: detection recall bounds end-to-end field accuracy, and where the errors come from.

Numbers are illustrative teaching shapes, not any company's reported results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_ocr_pipeline_error.png"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Left: end-to-end field F1 as the product of stage accuracies.
    det = np.linspace(0.80, 1.0, 100)
    for rec, c, lab in [(0.99, "C0", "recognition 0.99"), (0.95, "C1", "recognition 0.95"), (0.90, "C2", "recognition 0.90")]:
        ax1.plot(det, det * rec * 0.97, lw=2, color=c, label=f"{lab}, KV extraction 0.97")
    ax1.set_xlabel("text-detection recall")
    ax1.set_ylabel("end-to-end field accuracy")
    ax1.set_title("A pipeline multiplies: the weakest stage caps the system")
    ax1.axhline(0.90, color="k", ls="--", lw=1)
    ax1.text(0.805, 0.905, "0.90 SLA", fontsize=8)
    ax1.legend(fontsize=8, loc="lower right")

    # Right: where the residual errors live, by document condition.
    conds = ["clean scan", "phone photo", "low light", "crumpled /\nskewed", "handwriting"]
    detection = np.array([0.5, 1.4, 2.6, 4.1, 6.0])
    recognition = np.array([0.8, 2.0, 3.4, 4.0, 14.0])
    layout = np.array([0.6, 1.2, 1.4, 3.2, 3.0])
    x = np.arange(len(conds))
    ax2.bar(x, detection, 0.6, label="detection miss", color="C0", edgecolor="white", linewidth=2)
    ax2.bar(x, recognition, 0.6, bottom=detection, label="recognition error", color="C1", edgecolor="white", linewidth=2)
    ax2.bar(x, layout, 0.6, bottom=detection + recognition, label="layout / association error", color="C2", edgecolor="white", linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conds, fontsize=8)
    ax2.set_ylabel("field error rate (%)")
    ax2.set_title("Error mix changes with capture condition; report by condition")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
