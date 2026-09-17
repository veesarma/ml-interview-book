"""Classifier-free guidance on a toy conditional DDPM, and DDIM step-count quality.

Writes docs/assets/figures/part09_guidance.png.  Run:  python figures/part09_guidance.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mlbook.generative import ddpm as D  # noqa: E402
from mlbook.generative.toy_data import distance_to_mixture_modes, gaussian_mixture_2d_labeled  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part09_guidance.png"
torch.set_num_threads(1)
N_MODES = 6
SCALES = (0.0, 1.0, 3.0, 8.0)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    x, y = gaussian_mixture_2d_labeled(4096, n_modes=N_MODES, radius=2.0, std=0.15)
    sched = D.NoiseSchedule(D.cosine_alpha_bar_schedule(200))
    model = D.EpsMLP(d_x=2, n_classes=N_MODES, d_hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(4000):
        idx = torch.randint(0, x.shape[0], (256,))
        loss = D.ddpm_loss(model, sched, x[idx], y[idx], p_uncond=0.15)
        opt.zero_grad(); loss.backward(); opt.step()

    fig = plt.figure(figsize=(14, 6.0), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85], hspace=0.40, wspace=0.28)

    target = torch.zeros(600, dtype=torch.long)     # condition on mode 0, centred at (2, 0)
    spreads = []
    for i, w in enumerate(SCALES):
        ax = fig.add_subplot(gs[0, i])
        s = D.sample_ddim(model, sched, n=600, d=2, n_steps=50, y=target, guidance_scale=w)
        ax.scatter(x[:800, 0], x[:800, 1], s=3, alpha=0.18, color="#999999")
        ax.scatter(s[:, 0], s[:, 1], s=5, alpha=0.6, color=colors[1])
        ax.scatter([2.0], [0.0], marker="x", s=60, color="black")
        spreads.append((s - s.mean(dim=0)).norm(dim=1).mean().item())
        ax.set_title(f"guidance scale w = {w}", fontsize=10)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(SCALES, spreads, "o-", color=colors[3])
    ax.set_xlabel("guidance scale w"); ax.set_ylabel("mean distance to the sample mean")
    ax.set_title("Guidance buys fidelity by spending diversity", fontsize=10)

    ax = fig.add_subplot(gs[1, 2:])
    steps = [2, 5, 10, 20, 50, 100, 200]
    med = []
    for n in steps:
        s = D.sample_ddim(model, sched, n=400, d=2, n_steps=n)
        med.append(distance_to_mixture_modes(s, n_modes=N_MODES).median().item())
    ax.plot(steps, med, "o-", color=colors[0], label="DDIM")
    anc = D.sample_ancestral(model, sched, n=400, d=2)
    ax.axhline(distance_to_mixture_modes(anc, n_modes=N_MODES).median().item(),
               color=colors[2], linestyle="--", label="ancestral DDPM, all 200 steps")
    ax.set_xscale("log"); ax.set_xlabel("DDIM sampling steps")
    ax.set_ylabel("median distance to nearest mode")
    ax.set_title("DDIM buys back most of the quality in 10 to 20 steps", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    for ax in fig.axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
