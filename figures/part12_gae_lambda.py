"""Bias and variance of the GAE advantage estimate at the start state as lambda varies, with a biased critic.

Writes docs/assets/figures/part12_gae_lambda.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.rl.dynamic_programming import policy_evaluation, q_from_v  # noqa: E402
from mlbook.rl.envs import GridWorld  # noqa: E402
from mlbook.rl.gae import compute_gae  # noqa: E402
from mlbook.rl.mc_td import generate_episode  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_gae_lambda.png"
GAMMA, N_EPISODES = 0.95, 3000
LAMBDAS = np.linspace(0.0, 1.0, 11)


def main() -> None:
    env = GridWorld()
    pi = np.full((16, 4), 0.25)
    V_true = policy_evaluation(env.P, env.R, pi, GAMMA)  # (S,)
    Q_true = q_from_v(env.P, env.R, V_true, GAMMA)  # (S, A)
    rng = np.random.default_rng(0)
    critics = {"exact critic V = V^π": V_true, "biased critic V = 0.7·V^π": 0.7 * V_true}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), facecolor="white")
    episodes = [generate_episode(env, pi, rng) for _ in range(N_EPISODES)]
    for name, V_hat in critics.items():
        bias, var = np.zeros(len(LAMBDAS)), np.zeros(len(LAMBDAS))
        for j, lam in enumerate(LAMBDAS):
            est, truth = [], []
            for states, actions, rewards in episodes:
                values = V_hat[states]  # (T,)
                dones = np.zeros(len(rewards))
                dones[-1] = 1.0 if env.is_terminal(env.state) else 0.0
                last_value = 0.0 if dones[-1] else V_hat[env.state]
                adv, _ = compute_gae(np.asarray(rewards), values, dones, last_value, GAMMA, lam)
                est.append(adv[0])
                truth.append(Q_true[states[0], actions[0]] - V_true[states[0]])
            est, truth = np.asarray(est), np.asarray(truth)
            bias[j] = np.mean(est - truth)
            var[j] = np.var(est - truth)
        axes[0].plot(LAMBDAS, bias, marker="o", ms=3, label=name)
        axes[1].plot(LAMBDAS, var, marker="o", ms=3, label=name)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("bias  E[Â₀ − A(s₀,a₀)]")
    axes[0].set_title("Bias vanishes as λ→1 (no bootstrapping) ...", fontsize=9.5, loc="left")
    axes[1].set_ylabel("variance of Â₀ − A(s₀,a₀)")
    axes[1].set_title("... while variance grows with λ (more sampled rewards)", fontsize=9.5, loc="left")
    for ax in axes:
        ax.set_xlabel("λ")
        ax.grid(color="#eeeeee")
        ax.legend(fontsize=8, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
