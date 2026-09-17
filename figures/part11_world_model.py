"""A latent world model: training curve, imagined rollout vs truth, and CEM planning.

Writes docs/assets/figures/part11_world_model.png
Run from the repository root:  python figures/part11_world_model.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.perception.world_model import (  # noqa: E402
    LatentWorldModel,
    PointMassDynamics,
    cem_plan,
    random_shooting_plan,
    world_model_loss,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_world_model.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    np.random.seed(0)
    env = PointMassDynamics(dt=0.2)
    s0 = np.random.uniform(-1, 1, size=(512, 4))
    a_np = np.random.uniform(-1, 1, size=(512, 5, 2))
    obs_all = torch.tensor(env.rollout(s0, a_np), dtype=torch.float32)
    act_all = torch.tensor(a_np, dtype=torch.float32)

    model = LatentWorldModel(4, 2, latent_dim=8, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(400):
        idx = torch.randint(0, 512, (128,))
        loss = world_model_loss(model, obs_all[idx], act_all[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), facecolor="white")

    # ---- 1. the two loss terms -------------------------------------------------------
    ax = axes[0]
    ax.plot(losses, color=colors[0], lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("gradient step", fontsize=9)
    ax.set_ylabel("reconstruction + latent consistency", fontsize=8.5)
    ax.set_title("training the transition to chase the encoder", fontsize=9.8, loc="left")
    ax.grid(color="#eeeeee", lw=0.5, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- 2. imagined vs true rollout -------------------------------------------------
    ax = axes[1]
    start = np.array([[0.0, 0.0, 0.3, -0.2]])
    actions = np.random.uniform(-1, 1, size=(1, 10, 2))
    true = env.rollout(start, actions)[0]
    with torch.no_grad():
        z0 = model.encode(torch.tensor(start, dtype=torch.float32))
        imagined = model.decode(model.rollout(z0, torch.tensor(actions, dtype=torch.float32)))[0].numpy()
    ax.plot(true[:, 0], true[:, 1], "-o", ms=4, color="#444444", lw=2.0, label="true rollout")
    ax.plot(imagined[:, 0], imagined[:, 1], "--s", ms=4, color=colors[2], lw=2.0, label="imagined in latent space")
    for t in range(true.shape[0]):
        ax.plot([true[t, 0], imagined[t, 0]], [true[t, 1], imagined[t, 1]], color=colors[3], lw=0.7)
    ax.set_title("10 steps of open-loop imagination, never grounded", fontsize=9.8, loc="left")
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)
    ax.legend(fontsize=7.8, frameon=False)
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- 3. planning in the latent space ---------------------------------------------
    ax = axes[2]
    goal = torch.tensor([1.0, 1.0])
    start = np.array([[0.0, 0.0, 0.0, 0.0]])

    def cost_fn(decoded):
        return (decoded[:, -1, :2] - goal).norm(dim=-1)

    z0 = model.encode(torch.tensor(start, dtype=torch.float32))
    plans = {
        "random shooting, 32 samples": random_shooting_plan(model, z0, cost_fn, 8, 2, n_samples=32),
        "random shooting, 512 samples": random_shooting_plan(model, z0, cost_fn, 8, 2, n_samples=512),
        "CEM, 128 x 4 iters": cem_plan(model, z0, cost_fn, 8, 2, n_samples=128, n_elite=16, n_iters=4),
    }
    for (label, plan), c in zip(plans.items(), [colors[7], colors[1], colors[2]]):
        traj = env.rollout(start, plan.numpy()[None])[0]
        err = np.linalg.norm(traj[-1, :2] - goal.numpy())
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3.5, color=c, lw=1.9, label=f"{label} ({err:.2f} m off)")
    ax.scatter([1.0], [1.0], marker="*", s=220, color="#b00020", zorder=5)
    ax.text(0.97, 0.96, "goal", fontsize=8.5, ha="right", va="top", color="#b00020")
    ax.scatter([0], [0], marker="s", s=50, color="black", zorder=5)
    ax.set_title("plans found in latent space, executed in the real system", fontsize=9.8, loc="left")
    ax.set_xlabel("x", fontsize=9)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
