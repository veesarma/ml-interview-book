"""GAN training on a 2-D mixture: samples over time, loss curves, discriminator landscape.

Writes docs/assets/figures/part09_gan_toy.png.  Run:  python figures/part09_gan_toy.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.generative import gan as G  # noqa: E402
from mlbook.generative.toy_data import gaussian_mixture_2d  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part09_gan_toy.png"
torch.set_num_threads(1)
SNAPSHOTS = (0, 200, 1000, 4000)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    x = gaussian_mixture_2d(4096, n_modes=8, radius=2.0, std=0.12)
    gen = G.Generator(d_z=2, d_x=2, d_hidden=96)
    disc = G.Discriminator(d_x=2, d_hidden=96)
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-3, betas=(0.5, 0.999))

    snaps, d_hist, g_hist = {}, [], []
    for step in range(SNAPSHOTS[-1] + 1):
        if step in SNAPSHOTS:
            with torch.no_grad():
                snaps[step] = gen(torch.randn(800, 2)).numpy()
        idx = torch.randint(0, x.shape[0], (128,))
        d_l, g_l = G.gan_train_step(gen, disc, opt_g, opt_d, x[idx], d_z=2)
        d_hist.append(d_l); g_hist.append(g_l)

    fig = plt.figure(figsize=(14, 6.4), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85], hspace=0.38, wspace=0.28)

    for i, step in enumerate(SNAPSHOTS):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(x[:800, 0], x[:800, 1], s=4, alpha=0.25, color="#999999", label="data")
        ax.scatter(snaps[step][:, 0], snaps[step][:, 1], s=4, alpha=0.6, color=colors[1], label="G(z)")
        ax.set_title(f"step {step}", fontsize=10)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper right", frameon=False)

    ax = fig.add_subplot(gs[1, :2])
    w = 50
    smooth = lambda v: np.convolve(v, np.ones(w) / w, mode="valid")  # noqa: E731
    ax.plot(smooth(d_hist), color=colors[0], linewidth=1.2, label="discriminator loss")
    ax.plot(smooth(g_hist), color=colors[1], linewidth=1.2, label="generator loss (non-saturating)")
    ax.axhline(np.log(4), color="#888888", linestyle="--", linewidth=1.0)
    ax.text(len(d_hist) * 0.75, np.log(4) + 0.05, "log 4: D at chance", fontsize=8, color="#555555")
    ax.set_xlabel("training step"); ax.set_ylabel("loss (50-step moving average)")
    ax.set_title("Neither loss goes to zero. Read the samples, not the loss.", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 2:])
    g = np.linspace(-3.2, 3.2, 160)
    gx, gy = np.meshgrid(g, g)
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        d_prob = torch.sigmoid(disc(grid)).numpy().reshape(gx.shape)
    im = ax.contourf(gx, gy, d_prob, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
    ax.scatter(x[:600, 0], x[:600, 1], s=3, alpha=0.5, color="black")
    fig.colorbar(im, ax=ax, label="D(x) after training")
    ax.set_title("D(x) hovers near 0.5 on the data: no free win left", fontsize=10)
    ax.set_aspect("equal")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
