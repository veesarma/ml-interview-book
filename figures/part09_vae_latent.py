"""VAE on a 2-D mixture: latent posteriors, decoded samples, and the rate-distortion trade-off.

Writes docs/assets/figures/part09_vae_latent.png.  Run:  python figures/part09_vae_latent.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mlbook.generative import vae as V  # noqa: E402
from mlbook.generative.toy_data import gaussian_mixture_2d_labeled  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part09_vae_latent.png"
torch.set_num_threads(1)


def train(x: torch.Tensor, sigma: float, steps: int = 1500) -> tuple[V.VAE, float, float]:
    torch.manual_seed(0)
    model = V.VAE(d_x=2, d_z=2, d_hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(steps):
        x_hat, mu, logvar = model(x)
        loss, recon, kl = V.vae_loss(x, x_hat, mu, logvar, sigma_dec=sigma)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mse = ((x - model(x)[0]) ** 2).sum(dim=1).mean().item()
    return model, kl.item(), mse


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x, y = gaussian_mixture_2d_labeled(1500, n_modes=6, radius=2.0, std=0.15)
    model, _, _ = train(x, sigma=0.1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), facecolor="white")

    # panel 1: data
    for k in range(6):
        m = y == k
        axes[0].scatter(x[m, 0], x[m, 1], s=5, alpha=0.6, color=colors[k % 10])
    axes[0].set_title("Data: 6 Gaussians on a circle", fontsize=10)
    axes[0].set_xlabel("$x_1$"); axes[0].set_ylabel("$x_2$")

    # panel 2: encoder means in latent space
    with torch.no_grad():
        mu, logvar = model.encode(x)
    for k in range(6):
        m = y == k
        axes[1].scatter(mu[m, 0], mu[m, 1], s=5, alpha=0.6, color=colors[k % 10])
    circle = plt.Circle((0, 0), 1.0, fill=False, color="black", linestyle="--", linewidth=1.0)
    axes[1].add_patch(circle)
    axes[1].set_title(r"Latent posterior means $\mu(x)$ ($\sigma_{dec}=0.1$)", fontsize=10)
    axes[1].set_xlabel("$z_1$"); axes[1].set_ylabel("$z_2$")
    axes[1].text(0.02, 0.02, "dashed circle: 1 sd of the N(0, I) prior",
                 transform=axes[1].transAxes, fontsize=7.5, color="#555555")

    # panel 3: rate-distortion as sigma_dec varies
    sigmas = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    kls, mses = [], []
    for s in sigmas:
        _, kl, mse = train(x, sigma=s, steps=900)
        kls.append(kl); mses.append(mse)
    axes[2].plot(kls, mses, "o-", color=colors[3], linewidth=1.6)
    for s, kl, mse in zip(sigmas, kls, mses):
        axes[2].annotate(f"$\\sigma$={s}", (kl, mse), textcoords="offset points",
                         xytext=(6, 6), fontsize=7.5)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("rate: KL(q(z|x) || p(z)) in nats")
    axes[2].set_ylabel("distortion: mean squared error (log scale)")
    axes[2].set_title("Rate-distortion: the decoder variance picks the point", fontsize=10)
    axes[2].axvline(1.7918, color="#888888", linestyle=":", linewidth=1.0)
    axes[2].text(0.30, 0.45, "log 6 = 1.79 nats:\nenough to name the mode", transform=axes[2].transAxes,
                 fontsize=7.5, color="#555555")

    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_aspect("equal"); axes[1].set_aspect("equal")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
