"""Taxonomy of deep generative models: likelihood-based, implicit, score/flow-based.

Writes docs/assets/figures/part09_taxonomy.png.  Run:  python figures/part09_taxonomy.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part09_taxonomy.png"

# (x, y, w, h, title, body, colour index)
BOXES = [
    (0.02, 0.60, 0.30, 0.30, "Likelihood-based (explicit)",
     "Autoregressive: exact log p(x)\nVAE: ELBO lower bound\nNormalising flows: exact,\ninvertible, log-det", 0),
    (0.35, 0.60, 0.30, 0.30, "Implicit",
     "GANs: sample from G(z),\nno density at all.\nTrained by a discriminator\nthat estimates a divergence", 1),
    (0.68, 0.60, 0.30, 0.30, "Score / ODE-based",
     "Diffusion: learn the score\nof noised marginals.\nFlow matching: learn a\nvelocity field", 2),
]

LEAVES = [
    (0.02, 0.30, 0.30, 0.20, "What you get", "Density estimates, compression,\nanomaly scores. Samples are\nblurry when the decoder is Gaussian", 0),
    (0.35, 0.30, 0.30, 0.20, "What you get", "Sharp samples, one network\nevaluation. Unstable training,\nmode dropping, no likelihood", 1),
    (0.68, 0.30, 0.30, 0.20, "What you get", "Sharp samples and stable\nregression training. Many network\nevaluations per sample", 2),
]


def box(ax, x, y, w, h, title, body, color, fontsize=8.5):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                           linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.10)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=fontsize + 1.0,
            fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=fontsize, color="#222222")


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.975, "Three ways to fit a distribution you can only sample from",
            ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.935, "the split is how the model relates to the density p(x)",
            ha="center", va="top", fontsize=9.5, color="#555555")

    for x, y, w, h, t, b, ci in BOXES:
        box(ax, x, y, w, h, t, b, colors[ci])
    for x, y, w, h, t, b, ci in LEAVES:
        box(ax, x, y, w, h, t, b, colors[ci], fontsize=8.0)
    for x in (0.17, 0.50, 0.83):
        ax.add_patch(FancyArrowPatch((x, 0.60), (x, 0.505), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.1, color="#777777"))

    # the production stack: where each family actually sits today
    stack = FancyBboxPatch((0.02, 0.04), 0.96, 0.18, boxstyle="round,pad=0.014",
                           linewidth=1.4, edgecolor="#444444", facecolor="#f2f2f2")
    ax.add_patch(stack)
    ax.text(0.5, 0.195, "How a 2026 text-to-image or text-to-video system combines them",
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.145,
            "VAE or VQ-GAN encoder compresses pixels to latents  (likelihood-based, plus a GAN loss for sharpness)\n"
            "diffusion or flow matching models the latent distribution, conditioned on text  (score / ODE-based)\n"
            "VAE decoder maps the sampled latent back to pixels",
            ha="center", va="top", fontsize=9, color="#222222")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
