"""The learning-rate schedules that appear in LLM training reports."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.optim.schedules import inverse_sqrt, step_decay, warmup_cosine, warmup_stable_decay

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_lr_schedules.png"


def main() -> None:
    T = 10_000
    steps = np.arange(T)
    peak = 3e-4
    curves = {
        "warmup + cosine → 10% (Llama-style)": [warmup_cosine(s, 500, T, peak, min_lr=0.1 * peak) for s in steps],
        "warmup-stable-decay (WSD)": [warmup_stable_decay(s, 500, 8000, 1500, peak, min_lr=0.0) for s in steps],
        "inverse sqrt (Vaswani 2017)": [inverse_sqrt(s, 500, peak) for s in steps],
        "step decay ×0.1 (ResNet-style)": [step_decay(s, peak, 3000) for s in steps],
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, lr), c in zip(curves.items(), ["C0", "C1", "C2", "C3"]):
        ax.plot(steps, lr, lw=2, color=c, label=name)
    ax.axvspan(0, 500, color="0.9", label="warmup (500 steps)")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedules (peak 3e-4, 10k steps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
