"""Rectified-flow trajectories against a diffusion (probability-flow) trajectory, and step counts.

Writes docs/assets/figures/part09_flow_trajectories.png.
Run:  python figures/part09_flow_trajectories.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mlbook.generative import ddpm as D  # noqa: E402
from mlbook.generative import flow_matching as FM  # noqa: E402
from mlbook.generative.toy_data import distance_to_mixture_modes, gaussian_mixture_2d  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part09_flow_trajectories.png"
torch.set_num_threads(1)
N_MODES = 6


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    x = gaussian_mixture_2d(4096, n_modes=N_MODES, radius=2.0, std=0.12)

    flow = FM.VelocityMLP(d_x=2, d_hidden=128)
    opt = torch.optim.Adam(flow.parameters(), lr=3e-3)
    for _ in range(4000):
        idx = torch.randint(0, x.shape[0], (256,))
        loss = FM.cfm_loss(flow, x[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    sched = D.NoiseSchedule(D.cosine_alpha_bar_schedule(200))
    eps_model = D.EpsMLP(d_x=2, n_classes=0, d_hidden=128)
    opt2 = torch.optim.Adam(eps_model.parameters(), lr=3e-3)
    for _ in range(4000):
        idx = torch.randint(0, x.shape[0], (256,))
        loss = D.ddpm_loss(eps_model, sched, x[idx])
        opt2.zero_grad(); loss.backward(); opt2.step()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor="white")

    # panel 1: flow-matching trajectories
    traj = FM.sample_euler(flow, n=40, d=2, n_steps=60, return_trajectory=True)  # (61, 40, 2)
    axes[0].scatter(x[:800, 0], x[:800, 1], s=3, alpha=0.18, color="#999999")
    for i in range(traj.shape[1]):
        axes[0].plot(traj[:, i, 0], traj[:, i, 1], linewidth=0.8, alpha=0.7, color=colors[0])
    axes[0].scatter(traj[0, :, 0], traj[0, :, 1], s=10, color="black", zorder=3, label="t = 0 (noise)")
    axes[0].scatter(traj[-1, :, 0], traj[-1, :, 1], s=10, color=colors[3], zorder=3, label="t = 1 (data)")
    axes[0].set_title("Rectified flow: near-straight paths", fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")

    # panel 2: diffusion probability-flow (DDIM, eta = 0) trajectories, recorded step by step
    torch.manual_seed(1)
    n_steps = 60
    taus = torch.linspace(0, sched.T - 1, n_steps).round().long()
    xt = torch.randn(40, 2)
    path = [xt.clone()]
    with torch.no_grad():
        for i in reversed(range(n_steps)):
            t = taus[i].repeat(40)
            ab = sched.alpha_bar[taus[i]]
            ab_prev = sched.alpha_bar[taus[i - 1]] if i > 0 else torch.tensor(1.0)
            e = eps_model(xt, t)
            x0_hat = (xt - (1 - ab).sqrt() * e) / ab.sqrt()
            xt = ab_prev.sqrt() * x0_hat + (1 - ab_prev).clamp(min=0).sqrt() * e
            path.append(xt.clone())
    path_t = torch.stack(path)
    axes[1].scatter(x[:800, 0], x[:800, 1], s=3, alpha=0.18, color="#999999")
    for i in range(path_t.shape[1]):
        axes[1].plot(path_t[:, i, 0], path_t[:, i, 1], linewidth=0.8, alpha=0.7, color=colors[1])
    axes[1].scatter(path_t[0, :, 0], path_t[0, :, 1], s=10, color="black", zorder=3)
    axes[1].scatter(path_t[-1, :, 0], path_t[-1, :, 1], s=10, color=colors[3], zorder=3)
    axes[1].set_title("Diffusion probability-flow ODE: curved paths", fontsize=10)

    for ax in axes[:2]:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    # panel 3: sample quality against number of function evaluations
    nfe = [1, 2, 4, 8, 16, 32, 64]
    fm_q, dd_q = [], []
    for n in nfe:
        fm_q.append(distance_to_mixture_modes(FM.sample_euler(flow, 400, 2, n), N_MODES).median().item())
        dd_q.append(distance_to_mixture_modes(D.sample_ddim(eps_model, sched, 400, 2, n), N_MODES).median().item())
    axes[2].plot(nfe, fm_q, "o-", color=colors[0], label="flow matching, Euler")
    axes[2].plot(nfe, dd_q, "s-", color=colors[1], label="diffusion, DDIM")
    axes[2].set_xscale("log", base=2); axes[2].set_yscale("log")
    axes[2].set_xlabel("network evaluations per sample")
    axes[2].set_ylabel("median distance to nearest mode")
    axes[2].set_title("Straight paths tolerate a coarse integrator", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        axes[2].spines[side].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
