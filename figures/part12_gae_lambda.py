"""How lambda trades bias against variance in the GAE advantage estimate.

The policy gradient is invariant to a state-dependent baseline, so the part of the
advantage error that matters is the part that varies WITH THE ACTION at a state. This
script measures, at the start state of the gridworld:

  * gradient-relevant bias: RMS over actions of  E[A_hat | s0, a] - A(s0, a),  after
    subtracting the policy-weighted mean over actions (a constant offset biases nothing);
  * variance of the per-episode estimate A_hat(s0, a0).

Both are measured against the exact advantage from dynamic programming, for an exact
critic and for a deliberately biased one.

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
from mlbook.rl.mc_td import sample_action  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_gae_lambda.png"
GAMMA, N_EPISODES = 0.95, 6000
LAMBDAS = np.linspace(0.0, 1.0, 11)


def rollout(env, pi, rng):
    """One episode. Returns (states (T,), actions (T,), rewards (T,), final_state, terminated)."""
    env.reset(rng)
    states, actions, rewards = [], [], []
    done = False
    while not done:
        s = env.state
        a = sample_action(pi, s, rng)
        _, r, done = env.step(a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
    return (
        np.asarray(states),  # (T,)
        np.asarray(actions),  # (T,)
        np.asarray(rewards, dtype=np.float64),  # (T,)
        env.state,
        env.is_terminal(env.state),
    )


def main() -> None:
    env = GridWorld()
    pi = np.full((16, 4), 0.25)  # (S, A) uniform random policy
    V_true = policy_evaluation(env.P, env.R, pi, GAMMA)  # (S,)
    Q_true = q_from_v(env.P, env.R, V_true, GAMMA)  # (S, A)
    start = env.to_index(env.start)
    A_true = Q_true[start] - V_true[start]  # (A,) exact advantage at the start state

    rng = np.random.default_rng(0)
    episodes = [rollout(env, pi, rng) for _ in range(N_EPISODES)]

    critics = {"exact critic V = V^pi": V_true, "biased critic V = 0.7 V^pi": 0.7 * V_true}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), facecolor="white")

    for name, V_hat in critics.items():
        bias, var = np.zeros(len(LAMBDAS)), np.zeros(len(LAMBDAS))
        for j, lam in enumerate(LAMBDAS):
            est = np.zeros(N_EPISODES)  # (N,) A_hat at t = 0
            a0 = np.zeros(N_EPISODES, dtype=int)  # (N,) first action
            for i, (states, actions, rewards, final_state, terminated) in enumerate(episodes):
                values = V_hat[states]  # (T,)
                dones = np.zeros(len(rewards))  # (T,)
                dones[-1] = 1.0 if terminated else 0.0
                last_value = 0.0 if terminated else float(V_hat[final_state])
                adv, _ = compute_gae(rewards, values, dones, last_value, GAMMA, lam)
                est[i] = adv[0]
                a0[i] = actions[0]
            err = np.array([est[a0 == a].mean() - A_true[a] for a in range(4)])  # (A,)
            centred = err - float(pi[start] @ err)  # remove the pure baseline offset
            bias[j] = float(np.sqrt(np.mean(centred**2)))
            var[j] = float(np.var(est - A_true[a0]))
        axes[0].plot(LAMBDAS, bias, marker="o", ms=3, label=name)
        axes[1].plot(LAMBDAS, var, marker="o", ms=3, label=name)

    axes[0].set_ylabel("gradient-relevant bias (RMS over actions)")
    axes[0].set_title("A wrong critic biases the gradient only when lambda < 1", fontsize=9.5, loc="left")
    axes[1].set_ylabel("variance of the per-episode estimate")
    axes[1].set_title("Variance climbs with lambda (more sampled rewards per estimate)", fontsize=9.5, loc="left")
    for ax in axes:
        ax.set_xlabel("lambda")
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
