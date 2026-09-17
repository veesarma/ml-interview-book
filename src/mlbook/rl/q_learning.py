"""SARSA, Q-learning, Expected SARSA and Double Q-learning on a tabular MDP (pure NumPy).

The four algorithms differ only in the bootstrap target:

* SARSA (on-policy):      ``r + gamma Q(s', a')``            with ``a' ~ behaviour policy``
* Expected SARSA:         ``r + gamma sum_a' pi(a'|s') Q(s', a')``
* Q-learning (off-policy): ``r + gamma max_a' Q(s', a')``
* Double Q-learning:       ``r + gamma Q_B(s', argmax_a' Q_A(s', a'))``  (and vice versa)

All use ``Q(s,a) <- Q(s,a) + alpha [target - Q(s,a)]`` and eps-greedy behaviour.
"""

from __future__ import annotations

import numpy as np


def epsilon_greedy_action(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    """Random action w.p. ``eps``, else ``argmax_a Q(s, a)`` with random tie-breaking."""
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    row = Q[s]  # (A,)
    best = np.flatnonzero(row == row.max())  # ties
    return int(rng.choice(best))


def sarsa(
    env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """On-policy TD control; returns ``Q`` (S, A)."""
    Q = np.zeros((env.n_states, env.n_actions))  # (S, A)
    for _ in range(n_episodes):
        env.reset(rng)
        s = env.state
        a = epsilon_greedy_action(Q, s, eps, rng)
        done = False
        while not done:
            _, r, done = env.step(a, rng)
            s2 = env.state
            a2 = epsilon_greedy_action(Q, s2, eps, rng)  # the action we WILL take
            target = r + (0.0 if env.is_terminal(s2) else gamma * Q[s2, a2])
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
    return Q


def expected_sarsa(
    env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """Like SARSA but the target averages over the eps-greedy policy instead of sampling ``a'``."""
    Q = np.zeros((env.n_states, env.n_actions))  # (S, A)
    A = env.n_actions
    for _ in range(n_episodes):
        env.reset(rng)
        done = False
        while not done:
            s = env.state
            a = epsilon_greedy_action(Q, s, eps, rng)
            _, r, done = env.step(a, rng)
            s2 = env.state
            probs = np.full(A, eps / A)  # (A,) eps-greedy distribution at s'
            probs[np.argmax(Q[s2])] += 1.0 - eps
            target = r + (0.0 if env.is_terminal(s2) else gamma * float(probs @ Q[s2]))
            Q[s, a] += alpha * (target - Q[s, a])
    return Q


def q_learning(
    env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """Off-policy TD control: ``Q <- Q + alpha [ r + gamma max_a' Q(s',a') - Q ]``; returns (S, A)."""
    Q = np.zeros((env.n_states, env.n_actions))  # (S, A)
    for _ in range(n_episodes):
        env.reset(rng)
        done = False
        while not done:
            s = env.state
            a = epsilon_greedy_action(Q, s, eps, rng)
            _, r, done = env.step(a, rng)
            s2 = env.state
            target = r + (0.0 if env.is_terminal(s2) else gamma * float(np.max(Q[s2])))
            Q[s, a] += alpha * (target - Q[s, a])
    return Q


def double_q_learning(
    env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """Two tables; select the argmax with one and evaluate it with the other (van Hasselt, 2010).

    Returns ``(Q_A + Q_B) / 2`` of shape (S, A).
    """
    QA = np.zeros((env.n_states, env.n_actions))  # (S, A)
    QB = np.zeros((env.n_states, env.n_actions))  # (S, A)
    for _ in range(n_episodes):
        env.reset(rng)
        done = False
        while not done:
            s = env.state
            a = epsilon_greedy_action(QA + QB, s, eps, rng)  # behave with the sum
            _, r, done = env.step(a, rng)
            s2 = env.state
            if rng.random() < 0.5:
                a_star = int(np.argmax(QA[s2]))  # choose with A ...
                target = r + (0.0 if env.is_terminal(s2) else gamma * QB[s2, a_star])  # ... evaluate with B
                QA[s, a] += alpha * (target - QA[s, a])
            else:
                a_star = int(np.argmax(QB[s2]))
                target = r + (0.0 if env.is_terminal(s2) else gamma * QA[s2, a_star])
                QB[s, a] += alpha * (target - QB[s, a])
    return 0.5 * (QA + QB)


def maximisation_bias_demo(n_trials: int, n_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Why ``max`` of noisy estimates is biased upward, and why double estimation is not.

    Ten actions all have true value 0; each estimate is the mean of ``n_samples``
    N(0, 1) rewards. Returns ``(E[max_a Q(a)], E[Q_B(argmax_a Q_A(a))])`` over trials:
    the first is positive, the second is ~0.
    """
    single, double = 0.0, 0.0
    for _ in range(n_trials):
        QA = rng.normal(size=(10, n_samples)).mean(axis=1)  # (A,) noisy estimates
        QB = rng.normal(size=(10, n_samples)).mean(axis=1)  # (A,) independent estimates
        single += float(QA.max())
        double += float(QB[np.argmax(QA)])
    return single / n_trials, double / n_trials
