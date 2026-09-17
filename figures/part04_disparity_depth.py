"""Disparity ↔ depth and the quadratic growth of depth error with range."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.vision.depth import depth_error_from_disparity_error, disparity_to_depth

OUT = "docs/assets/figures/part04_disparity_depth.png"


def main():
    f = 700.0  # px
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    d = np.linspace(1, 100, 400)
    for b in (0.12, 0.54, 1.2):
        ax1.plot(d, disparity_to_depth(d, f, b), label=f"baseline {b} m")
    ax1.set_xlabel("disparity (px)")
    ax1.set_ylabel("depth Z = f·b/d (m)")
    ax1.set_ylim(0, 120)
    ax1.set_title("hyperbolic: most of the disparity range covers near depths")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    Z = np.linspace(2, 120, 300)
    for b in (0.12, 0.54, 1.2):
        ax2.plot(Z, depth_error_from_disparity_error(Z, f, b, 0.5), label=f"baseline {b} m")
    ax2.set_xlabel("depth Z (m)")
    ax2.set_ylabel("depth error for 0.5 px disparity error (m)")
    ax2.set_title("|ΔZ| ≈ Z²·Δd/(f·b): error grows with Z²")
    ax2.set_ylim(0, 40)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
