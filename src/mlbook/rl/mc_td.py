"""Monte Carlo and temporal-difference prediction / control on a tabular MDP (pure NumPy).

All routines interact with an environment through ``env.reset(rng)`` /
``env.step(a, rng)`` and read the integer state from ``env.state``; observations
are ignored because these are tabular methods. Targets:

* MC:       ``G_t = r_{t+1} + gamma r_{t+2} + ...``               (unbiased, high variance)
* TD(0):    ``r_{t+1} + gamma V(s_{t+1})``                       (biased by V, low variance)
* n-step:   ``r_{t+1} + ... + gamma^{n-1} r_{t+n} + gamma^n V(s_{t+n})``
* TD(lambda): backward view with accumulating eligibility traces ``e``.
"""

from __future__ import annotations

import numpy as np


def sample_action(pi: np.ndarray, s: int, rng: np.random.Generator) -> int:
    """Draw ``a ~ pi(.|s)`` from a (S, A) table."""
    return int(rng.choice(pi.shape[1], p=pi[s]))


def generate_episode(env, pi: np.ndarray, rng: np.random.Generator) -> tuple[list[int], list[int], list[float]]:
    """Roll out ``pi`` once. Returns ``(states, actions, rewards)`` with ``len == T``."""
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
    return states, actions, rewards


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """``G_t = sum_{k>=0} gamma^k r_{t+k+1}`` computed backwards; (T,) -> (T,)."""
    G = np.zeros(len(rewards))  # (T,)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def mc_evaluation(env, pi: np.ndarray, gamma: float, n_episodes: int, rng: np.random.Generator) -> np.ndarray:
    """First-visit Monte Carlo estimate of ``V^pi``; returns (S,)."""
    S = env.n_states
    V, counts = np.zeros(S), np.zeros(S)  # (S,), (S,)
    for _ in range(n_episodes):
        states, _, rewards = generate_episode(env, pi, rng)
        G = discounted_returns(rewards, gamma)  # (T,)
        seen: set[int] = set()
        for t, s in enumerate(states):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G[t] - V[s]) / counts[s]  # incremental mean
    return V


def td0_evaluation(
    env, pi: np.ndarray, gamma: float, n_episodes: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """TD(0): ``V(s) <- V(s) + alpha [ r + gamma V(s') - V(s) ]``; returns (S,)."""
    V = np.zeros(env.n_states)  # (S,)
    for _ in range(n_episodes):
        env.reset(rng)
        done = False
        while not done:
            s = env.state
            a = sample_action(pi, s, rng)
            _, r, done = env.step(a, rng)
            s2 = env.state
            target = r + (0.0 if env.is_terminal(s2) else gamma * V[s2])
            V[s] += alpha * (target - V[s])  # TD error delta = target - V(s)
    return V


def n_step_td_evaluation(
    env, pi: np.ndarray, gamma: float, n: int, n_episodes: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """n-step TD prediction (Sutton & Barto, Ch. 7): bootstrap after ``n`` rewards; returns (S,)."""
    V = np.zeros(env.n_states)  # (S,)
    for _ in range(n_episodes):
        states, _, rewards = generate_episode(env, pi, rng)
        T = len(rewards)
        for t in range(T):
            end = min(t + n, T)
            G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
            if end < T:  # not yet terminal: bootstrap from V(s_{t+n})
                G += gamma**n * V[states[end]]
            V[states[t]] += alpha * (G - V[states[t]])
    return V


def td_lambda_evaluation(
    env, pi: np.ndarray, gamma: float, lam: float, n_episodes: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """TD(lambda), backward view with accumulating traces.

    Per step: ``delta = r + gamma V(s') - V(s)``, ``e <- gamma*lam*e; e[s] += 1``,
    ``V <- V + alpha * delta * e``. ``lam=0`` is TD(0); ``lam=1`` matches MC in the limit.
    """
    V = np.zeros(env.n_states)  # (S,)
    for _ in range(n_episodes):
        e = np.zeros(env.n_states)  # (S,) eligibility trace
        env.reset(rng)
        done = False
        while not done:
            s = env.state
            a = sample_action(pi, s, rng)
            _, r, done = env.step(a, rng)
            s2 = env.state
            delta = r + (0.0 if env.is_terminal(s2) else gamma * V[s2]) - V[s]
            e = gamma * lam * e  # decay every state's credit
            e[s] += 1.0  # the state just visited earns full credit
            V += alpha * delta * e  # (S,) every recently visited state moves
    return V


def epsilon_soft(Q: np.ndarray, eps: float) -> np.ndarray:
    """Turn (S, A) action values into an eps-soft policy table (S, A)."""
    S, A = Q.shape
    pi = np.full((S, A), eps / A)  # (S, A)
    pi[np.arange(S), np.argmax(Q, axis=1)] += 1.0 - eps
    return pi


def mc_control(
    env, gamma: float, n_episodes: int, eps: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """On-policy first-visit MC control with an eps-soft policy; returns ``(Q (S,A), pi (S,A))``."""
    Q = np.zeros((env.n_states, env.n_actions))  # (S, A)
    counts = np.zeros_like(Q)  # (S, A)
    pi = epsilon_soft(Q, eps)  # (S, A)
    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(env, pi, rng)
        G = discounted_returns(rewards, gamma)  # (T,)
        seen: set[tuple[int, int]] = set()
        for t, (s, a) in enumerate(zip(states, actions)):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s, a] += 1
            Q[s, a] += (G[t] - Q[s, a]) / counts[s, a]
        pi = epsilon_soft(Q, eps)
    return Q, pi
