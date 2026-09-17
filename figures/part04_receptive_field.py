"""Theoretical receptive field growth for common stacks, and the effective RF (Gaussian-like)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.vision.conv import receptive_field

OUT = "docs/assets/figures/part04_receptive_field.png"


def rf_curve(layers):
    out = []
    for i in range(1, len(layers) + 1):
        out.append(receptive_field(layers[:i])[0])
    return out


def main():
    stacks = {
        "3×3, stride 1 (VGG)": [(3, 1, 1)] * 12,
        "3×3 with stride-2 every 3rd layer": [(3, 1, 1), (3, 1, 1), (3, 2, 1)] * 4,
        "3×3 dilated 1,2,4,8 repeated": [(3, 1, d) for d in (1, 2, 4, 8)] * 3,
        "7×7 s2 stem then 3×3 s1": [(7, 2, 1)] + [(3, 1, 1)] * 11,
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for name, layers in stacks.items():
        ax1.plot(range(1, 13), rf_curve(layers), marker="o", ms=3, label=name)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("theoretical receptive field (px)")
    ax1.set_title("RF grows linearly per layer, ×jump after each stride")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    # effective RF: sum of L random 3x3 uniform kernels ≈ Gaussian by CLT
    rng = np.random.default_rng(0)
    L = 8
    field = np.zeros((1, 1))
    field[0, 0] = 1.0
    for _ in range(L):
        k = np.ones((3, 3)) / 9.0
        field = np.pad(field, 1)
        new = np.zeros_like(field)
        for u in range(3):
            for v in range(3):
                new += k[u, v] * np.roll(np.roll(field, u - 1, 0), v - 1, 1)
        field = new
    ax2.imshow(field, cmap="viridis")
    ax2.set_title(f"effective RF after {L} 3×3 layers (theoretical = {2*L+1}px box)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
