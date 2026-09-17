"""Focal loss vs cross-entropy for several γ, and its gradient magnitude."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.detection.focal_loss import focal_loss_grad_numpy

OUT = "docs/assets/figures/part04_focal_loss.png"


def main():
    p = np.linspace(0.01, 0.999, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for g in (0.0, 0.5, 1.0, 2.0, 5.0):
        ax1.plot(p, -((1 - p) ** g) * np.log(p), label=f"γ = {g}" + ("  (cross-entropy)" if g == 0 else ""))
    ax1.set_xlabel("p_t  (probability of the true class)")
    ax1.set_ylabel("loss")
    ax1.set_ylim(0, 5)
    ax1.set_title("FL = −(1 − p_t)^γ log p_t")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    z = np.linspace(-6, 6, 400)
    y = np.ones_like(z, dtype=np.int64)
    for g in (0.0, 1.0, 2.0):
        ax2.plot(1 / (1 + np.exp(-z)), -focal_loss_grad_numpy(z, y, alpha=1.0, gamma=g), label=f"γ = {g}")
    ax2.set_xlabel("p = σ(z) for a positive example")
    ax2.set_ylabel("−∂FL/∂z  (gradient magnitude on the logit)")
    ax2.set_title("γ > 0 kills the gradient of well-classified examples")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
