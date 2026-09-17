"""Behavioural cloning vs DAgger on the corridor toy, plus the T vs T^2 error-growth argument.

Writes docs/assets/figures/part12_bc_vs_dagger.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.rl.behavioral_cloning import TabularPolicy, collect_expert_data  # noqa: E402
from mlbook.rl.dagger import dagger, make_tabular_learner  # noqa: E402
from mlbook.rl.envs import CorridorEnv, run_episode  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_bc_vs_dagger.png"
SEEDS, EPISODES_PER_ITER, N_ITERS = 8, 20, 8


def mean_return(env, act, rng, n=100):
    return float(np.mean([run_episode(env, act, rng) for _ in range(n)]))


def main() -> None:
    env = CorridorEnv()
    bc_curve = np.zeros((SEEDS, N_ITERS))  # BC with the same total data budget as DAgger at each iteration
    dg_curve = np.zeros((SEEDS, N_ITERS))
    expert = np.zeros(SEEDS)
    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        expert[s] = mean_return(env, env.expert_action, rng)
        for i in range(N_ITERS):
            obs, acts = collect_expert_data(env, env.expert_action, EPISODES_PER_ITER * (i + 1), rng)
            bc = TabularPolicy(3, 0.05, rng)
            bc.fit(obs, acts)
            bc_curve[s, i] = mean_return(env, bc.act, rng)
        _, hist = dagger(env, env.expert_action, make_tabular_learner(3, 0.05, rng), N_ITERS, EPISODES_PER_ITER, rng, beta0=0.5)
        dg_curve[s] = hist
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), facecolor="white")
    x = np.arange(1, N_ITERS + 1) * EPISODES_PER_ITER
    for curve, name in ((bc_curve, "Behavioural cloning (more expert demos)"), (dg_curve, "DAgger (learner rollouts, expert labels)")):
        axes[0].plot(x, curve.mean(0), marker="o", ms=3, label=name)
        axes[0].fill_between(x, curve.mean(0) - curve.std(0), curve.mean(0) + curve.std(0), alpha=0.15)
    axes[0].axhline(expert.mean(), color="black", ls="--", lw=1, label=f"expert ({expert.mean():.1f})")
    axes[0].set_xlabel("labelled episodes in the dataset")
    axes[0].set_ylabel("return of the learned policy (max 40)")
    axes[0].set_title("Corridor toy: same label budget, different state distribution", fontsize=9.5, loc="left")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")

    T = np.arange(1, 101)
    eps = 0.02
    axes[1].plot(T, eps * T**2, label="BC bound: ε T²  (errors compound)")
    axes[1].plot(T, eps * T * 2, label="DAgger bound: O(ε T)  (learner sees its own states)")
    axes[1].set_xlabel("horizon T")
    axes[1].set_ylabel("expected cost gap to the expert (ε = 0.02)")
    axes[1].set_title("Why: Ross & Bagnell's T² vs T error growth", fontsize=9.5, loc="left")
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(color="#eeeeee")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
