"""How lambda trades bias against variance in the GAE advantage estimate.

The policy gradient is invariant to a state-dependent baseline, so the only part of the
advantage error that biases the gradient is the part that varies with the ACTION at a
state. This script measures both halves of the trade at the gridworld's start state:

  * bias, computed exactly by dynamic programming. With
        e(s)   = E[delta | s]   under pi        (S,)
        e0(a)  = E[delta_0 | s0, a]             (A,)
    the expected GAE estimate is
        E[A_hat | s0, a] = e0(a) + gamma*lam * P(.|s0,a)^T (I - gamma*lam*P_pi)^-1 e,
    and the plotted quantity is the RMS over actions after removing the policy-weighted
    mean (a constant offset changes no gradient).
  * variance of the per-episode estimate, from 4000 sampled episodes.

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
GAMMA, N_EPISODES = 0.95, 4000
LAMBDAS = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0])


def rollout(env, pi, rng):
    """One episode: (states (T,), actions (T,), rewards (T,), final_state, terminated)."""
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
    return np.asarray(states), np.asarray(actions), np.asarray(rewards, dtype=np.float64), env.state, env.is_terminal(env.state)


def exact_bias(env, pi, V_hat, A_true, start, lam):
    """RMS over actions of the action-centred error of E[A_hat | s0, a]; scalar."""
    P_pi = np.einsum("sa,sat->st", pi, env.P)  # (S, S)
    R_pi = np.sum(pi * env.R, axis=1)  # (S,)
    e = R_pi + GAMMA * P_pi @ V_hat - V_hat  # (S,) expected TD error in each state
    e0 = env.R[start] + GAMMA * env.P[start] @ V_hat - V_hat[start]  # (A,)
    resolvent = np.linalg.solve(np.eye(env.n_states) - GAMMA * lam * P_pi, e)  # (S,)
    expected_adv = e0 + GAMMA * lam * (env.P[start] @ resolvent)  # (A,)
    err = expected_adv - A_true  # (A,)
    centred = err - float(pi[start] @ err)  # drop the pure baseline offset
    return float(np.sqrt(np.mean(centred**2)))


def main() -> None:
    env = GridWorld()
    pi = np.full((env.n_states, env.n_actions), 0.25)  # (S, A) uniform random policy
    V_true = policy_evaluation(env.P, env.R, pi, GAMMA)  # (S,)
    Q_true = q_from_v(env.P, env.R, V_true, GAMMA)  # (S, A)
    start = env.to_index(env.start)
    A_true = Q_true[start] - V_true[start]  # (A,) exact advantage at the start state

    rng = np.random.default_rng(0)
    episodes = [rollout(env, pi, rng) for _ in range(N_EPISODES)]

    critics = {"exact critic V = V^pi": V_true, "biased critic V = 0.7 V^pi": 0.7 * V_true}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), facecolor="white")

    for name, V_hat in critics.items():
        bias = np.array([exact_bias(env, pi, V_hat, A_true, start, lam) for lam in LAMBDAS])
        var = np.zeros(len(LAMBDAS))
        for j, lam in enumerate(LAMBDAS):
            est = np.zeros(N_EPISODES)  # (N,)
            a0 = np.zeros(N_EPISODES, dtype=int)  # (N,)
            for i, (states, actions, rewards, final_state, terminated) in enumerate(episodes):
                values = V_hat[states]  # (T,)
                dones = np.zeros(len(rewards))  # (T,)
                dones[-1] = 1.0 if terminated else 0.0
                last_value = 0.0 if terminated else float(V_hat[final_state])
                adv, _ = compute_gae(rewards, values, dones, last_value, GAMMA, lam)
                est[i], a0[i] = adv[0], actions[0]
            var[j] = float(np.var(est - A_true[a0]))
        axes[0].plot(LAMBDAS, bias, marker="o", ms=3, label=name)
        axes[1].plot(LAMBDAS, var, marker="o", ms=3, label=name)

    axes[0].set_ylabel("bias of the gradient (RMS over actions)")
    axes[0].set_title("A wrong critic biases the gradient, and only when lambda < 1", fontsize=9.5, loc="left")
    axes[1].set_ylabel("variance of the per-episode estimate")
    axes[1].set_yscale("log")
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
