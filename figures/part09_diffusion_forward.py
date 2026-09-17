"""Forward diffusion on 2-D data, the closed-form marginal, and what the schedule does to SNR.

Writes docs/assets/figures/part09_diffusion_forward.png.
Run:  python figures/part09_diffusion_forward.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mlbook.generative import ddpm as D  # noqa: E402
from mlbook.generative.toy_data import gaussian_mixture_2d  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part09_diffusion_forward.png"
torch.set_num_threads(1)
T = 1000
STEPS = (0, 200, 400, 600, 999)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    x0 = gaussian_mixture_2d(1200, n_modes=8, radius=2.0, std=0.12)
    sched = D.NoiseSchedule(D.linear_beta_schedule(T))
    cos = D.NoiseSchedule(D.cosine_alpha_bar_schedule(T))

    fig = plt.figure(figsize=(14, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.25)

    eps = torch.randn_like(x0)                       # one fixed noise draw, so the panels are a path
    for i, t in enumerate(STEPS):
        ax = fig.add_subplot(gs[0, i])
        xt = D.q_sample(sched, x0, torch.full((x0.shape[0],), t, dtype=torch.long), eps)
        ax.scatter(xt[:, 0], xt[:, 1], s=4, alpha=0.5, color=colors[0])
        ab = sched.alpha_bar[t].item()
        ax.set_title(f"t = {t}\n" + r"$\bar\alpha_t$ = " + f"{ab:.3f}", fontsize=9)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(sched.alpha_bar, color=colors[0], label="linear schedule")
    ax.plot(cos.alpha_bar, color=colors[1], label="cosine schedule")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\bar\alpha_t$ (signal retained)")
    ax.set_title("Cosine keeps signal alive through the middle of the trajectory", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(sched.snr(), color=colors[0], label="linear")
    ax.plot(cos.snr(), color=colors[1], label="cosine")
    zero = D.NoiseSchedule(D.enforce_zero_terminal_snr(D.linear_beta_schedule(T)))
    ax.plot(zero.snr(), color=colors[2], linestyle="--", label="linear, zero terminal SNR")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(r"SNR = $\bar\alpha_t / (1-\bar\alpha_t)$")
    ax.set_title("SNR on a log scale. Watch the last step.", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 4])
    ax.axis("off")
    ax.text(0.0, 1.0,
            "Terminal SNR\n\n"
            f"linear:  {sched.snr()[-1]:.2e}\n"
            f"cosine:  {cos.snr()[-1]:.2e}\n"
            f"rescaled: {zero.snr()[-1]:.2e}\n\n"
            "A non-zero terminal SNR means\nthe model never trained on the\npure noise it is handed at\nsampling time. That leaks the\nmean brightness of the training\nset into every sample.",
            va="top", ha="left", fontsize=8, color="#333333", transform=ax.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
