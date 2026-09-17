"""U-Net architecture diagram (three levels, skip connections) with tensor shapes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUT = "docs/assets/figures/part04_unet.png"


def main():
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6)
    ax.axis("off")
    enc = [("x  (B,1,H,W)", 0.2, 5.0, "#dbe7ff"), ("e1  (B,w,H,W)", 2.6, 5.0, "#c7d8ff"), ("e2  (B,2w,H/2,W/2)", 3.2, 3.4, "#a9c1ff"), ("e3  (B,4w,H/4,W/4)", 4.0, 1.8, "#88a9ff")]
    dec = [("d2  (B,2w,H/2,W/2)", 8.2, 3.4, "#a9c1ff"), ("d1  (B,w,H,W)", 9.2, 5.0, "#c7d8ff"), ("logits (B,K,H,W)", 11.5, 5.0, "#dbe7ff")]
    for label, x, y, c in enc + dec:
        ax.add_patch(patches.FancyBboxPatch((x, y - 0.35), 1.9, 0.7, boxstyle="round,pad=0.04", fc=c, ec="#1f2937"))
        ax.text(x + 0.95, y, label, ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(2.6, 5.0), xytext=(2.1, 5.0), arrowprops=dict(arrowstyle="->"))
    ax.text(2.35, 5.45, "DoubleConv", fontsize=7.5, ha="center")
    ax.annotate("", xy=(4.1, 3.75), xytext=(3.6, 4.65), arrowprops=dict(arrowstyle="->"))
    ax.text(3.3, 4.25, "maxpool ↓2\nDoubleConv", fontsize=7.5, ha="right")
    ax.annotate("", xy=(4.9, 2.15), xytext=(4.4, 3.05), arrowprops=dict(arrowstyle="->"))
    ax.text(4.0, 2.6, "maxpool ↓2\nDoubleConv", fontsize=7.5, ha="right")
    ax.annotate("", xy=(8.2, 3.05), xytext=(5.9, 2.15), arrowprops=dict(arrowstyle="->"))
    ax.text(7.0, 2.2, "ConvT ↑2, concat skip,\nDoubleConv", fontsize=7.5, ha="center", va="top")
    ax.annotate("", xy=(9.5, 4.65), xytext=(9.2, 3.75), arrowprops=dict(arrowstyle="->"))
    ax.text(9.7, 4.15, "ConvT ↑2, concat,\nDoubleConv", fontsize=7.5)
    ax.annotate("", xy=(11.5, 5.0), xytext=(11.1, 5.0), arrowprops=dict(arrowstyle="->"))
    ax.text(11.3, 5.45, "1×1 conv", fontsize=7.5, ha="center")
    # skips
    ax.annotate("", xy=(9.2, 5.15), xytext=(4.5, 5.15), arrowprops=dict(arrowstyle="->", color="#059669", lw=2, linestyle="--"))
    ax.annotate("", xy=(8.2, 3.55), xytext=(5.1, 3.55), arrowprops=dict(arrowstyle="->", color="#059669", lw=2, linestyle="--"))
    ax.text(6.85, 5.35, "skip: copy e1 (full-res edges)", color="#059669", fontsize=8.5, ha="center")
    ax.text(6.65, 3.75, "skip: copy e2", color="#059669", fontsize=8.5, ha="center")
    ax.set_title("TinyUNet: encoder loses resolution to gain context; skips give the decoder the lost detail back")
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
