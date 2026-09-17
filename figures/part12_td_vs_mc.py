"""MC vs TD(0) vs TD(lambda) prediction error at the start state as a function of episodes (20 seeds).

Writes docs/assets/figures/part12_td_vs_mc.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.rl.dynamic_programming import policy_evaluation  # noqa: E402
from mlbook.rl.envs import GridWorld  # noqa: E402
from mlbook.rl.mc_td import discounted_returns, generate_episode, mc_evaluation, td0_evaluation, td_lambda_evaluation  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_td_vs_mc.png"
GAMMA, SEEDS = 0.95, 10
CHECKPOINTS = [25, 50, 100, 200, 400, 800, 1600]


def main() -> None:
    env = GridWorld()
    pi = np.full((16, 4), 0.25)
    V_true = policy_evaluation(env.P, env.R, pi, GAMMA)
    start = env.to_index(env.start)
    # Every method uses the same constant step size, so the comparison is about the target
    # and not about the schedule. The 1/n variant of MC is shown for reference.
    methods = {
        "MC (first visit), α=0.05": lambda n, rng: mc_evaluation(env, pi, GAMMA, n, rng, alpha=0.05),
        "TD(0), α=0.05": lambda n, rng: td0_evaluation(env, pi, GAMMA, n, 0.05, rng),
        "TD(λ=0.8), α=0.05": lambda n, rng: td_lambda_evaluation(env, pi, GAMMA, 0.8, n, 0.05, rng),
        "MC with 1/n averaging": lambda n, rng: mc_evaluation(env, pi, GAMMA, n, rng),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="white")
    for name, fn in methods.items():
        errs = np.zeros((SEEDS, len(CHECKPOINTS)))  # (SEEDS, checkpoints)
        for s in range(SEEDS):
            for j, n in enumerate(CHECKPOINTS):
                V = fn(n, np.random.default_rng(s))
                errs[s, j] = np.sqrt(np.mean((V - V_true) ** 2))
        axes[0].errorbar(CHECKPOINTS, errs.mean(0), yerr=errs.std(0), label=name, marker="o", ms=3, capsize=2, lw=1.4)
    axes[0].set_xscale("log")
    axes[0].set_xticks(CHECKPOINTS)
    axes[0].set_xticklabels([str(c) for c in CHECKPOINTS], fontsize=8)
    axes[0].minorticks_off()
    axes[0].set_xlabel("episodes")
    axes[0].set_ylabel("RMS error of V vs DP (all states)")
    axes[0].set_title("Prediction error at equal step size, uniform random policy on GridWorld", fontsize=9.5, loc="left")
    axes[0].legend(fontsize=8, frameon=False)

    # Target distributions at the start state: MC return vs TD target with the converged V
    rng = np.random.default_rng(0)
    mc_targets, td_targets = [], []
    for _ in range(600):
        states, _, rewards = generate_episode(env, pi, rng)
        G = discounted_returns(rewards, GAMMA)
        mc_targets.append(G[0])
        s1 = states[1] if len(states) > 1 else start
        td_targets.append(rewards[0] + (0.0 if env.is_terminal(s1) else GAMMA * V_true[s1]) if len(states) > 1 else rewards[0])
    axes[1].hist(mc_targets, bins=40, alpha=0.6, label=f"MC target G₀  (sd={np.std(mc_targets):.2f})")
    axes[1].hist(td_targets, bins=40, alpha=0.6, label=f"TD target r₁+γV(s₁)  (sd={np.std(td_targets):.2f})")
    axes[1].axvline(V_true[start], color="black", lw=1, ls="--", label=f"V^π(start)={V_true[start]:.2f}")
    axes[1].set_xlabel("target value at the start state")
    axes[1].set_ylabel("count (600 episodes)")
    axes[1].set_title("Both targets are unbiased once V is right; TD's has far less variance", fontsize=9.5, loc="left")
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
