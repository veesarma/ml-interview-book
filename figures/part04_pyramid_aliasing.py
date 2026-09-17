"""Gaussian/Laplacian pyramid and aliasing: naive decimation vs blur-then-decimate."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.vision.image_ops import downsample2, gaussian_pyramid, laplacian_pyramid

OUT = "docs/assets/figures/part04_pyramid_aliasing.png"


def zone_plate(n=128, k=0.8):
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    return 0.5 + 0.5 * np.cos(k * (xx**2 + yy**2) / n)  # chirp: frequency rises with radius


def main():
    img = zone_plate()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), facecolor="white")
    gp = gaussian_pyramid(img, 4)
    lp = laplacian_pyramid(img, 4)
    for l in range(4):
        axes[0, l].imshow(gp[l], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0, l].set_title(f"G{l}  {gp[l].shape[0]}×{gp[l].shape[1]}")
        axes[0, l].axis("off")
    naive = img[::4, ::4]
    aa = downsample2(downsample2(img))
    axes[1, 0].imshow(naive, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title("stride-4 subsample (aliased)")
    axes[1, 1].imshow(aa, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 1].set_title("blur → ×2 → blur → ×2")
    axes[1, 2].imshow(lp[0], cmap="gray", interpolation="nearest")
    axes[1, 2].set_title("L0 = G0 − up(G1)")
    axes[1, 3].imshow(lp[1], cmap="gray", interpolation="nearest")
    axes[1, 3].set_title("L1 = G1 − up(G2)")
    for ax in axes[1]:
        ax.axis("off")
    fig.suptitle("Gaussian pyramid (top); aliasing from naive decimation vs anti-aliased, and Laplacian bands (bottom)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
