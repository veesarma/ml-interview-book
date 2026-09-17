"""Dynamic programming on an explicit tabular MDP (pure NumPy).

Everything here is tensor algebra on ``P`` of shape ``(S, A, S)`` and ``R`` of
shape ``(S, A)``:

* Bellman expectation backup  ``V <- sum_a pi(a|s) [ R(s,a) + gamma sum_s' P(s,a,s') V(s') ]``
* Bellman optimality backup    ``V <- max_a [ R(s,a) + gamma sum_s' P(s,a,s') V(s') ]``

Policies are stochastic tables ``pi`` of shape ``(S, A)`` (rows sum to one);
``greedy_policy`` returns the deterministic ``(S,)`` integer version.
"""

from __future__ import annotations

import numpy as np


def q_from_v(P: np.ndarray, R: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    """``Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') V(s')``.

    Args:
        P: (S, A, S) transition probabilities.
        R: (S, A) expected rewards.
        V: (S,) state values.
    Returns:
        (S, A) action values.
    """
    expected_next = P @ V  # (S, A, S) @ (S,) -> (S, A): sum over s'
    return R + gamma * expected_next  # (S, A)


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Deterministic policy ``pi(s) = argmax_a Q(s, a)``; (S, A) -> (S,) ints."""
    return np.argmax(Q, axis=1)  # (S,)


def one_hot_policy(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """(S,) integer actions -> (S, A) stochastic table with a single 1 per row."""
    pi = np.zeros((actions.shape[0], n_actions))  # (S, A)
    pi[np.arange(actions.shape[0]), actions] = 1.0
    return pi


def policy_evaluation(
    P: np.ndarray, R: np.ndarray, pi: np.ndarray, gamma: float, tol: float = 1e-10, max_iters: int = 10_000
) -> np.ndarray:
    """Iterate the Bellman expectation backup until the sup-norm change is < ``tol``.

    Args:
        P: (S, A, S); R: (S, A); pi: (S, A) with rows summing to one.
    Returns:
        (S,) ``V^pi``.
    """
    V = np.zeros(P.shape[0])  # (S,)
    for _ in range(max_iters):
        Q = q_from_v(P, R, V, gamma)  # (S, A)
        V_new = np.sum(pi * Q, axis=1)  # (S,) expectation over a ~ pi(.|s)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def policy_evaluation_exact(P: np.ndarray, R: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    """Closed form ``V = (I - gamma P_pi)^-1 R_pi`` (valid because gamma < 1 makes I - gamma P_pi invertible).

    ``P_pi[s, s'] = sum_a pi(a|s) P(s,a,s')`` (S, S); ``R_pi[s] = sum_a pi(a|s) R(s,a)`` (S,).
    """
    P_pi = np.einsum("sa,sat->st", pi, P)  # (S, S): contract the action axis with pi(a|s)
    R_pi = np.sum(pi * R, axis=1)  # (S,)
    S = P.shape[0]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)  # (S,)


def value_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float, tol: float = 1e-10, max_iters: int = 10_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Bellman optimality backups ``V <- max_a Q`` until convergence.

    Returns:
        ``(V_star (S,), pi_star (S,) ints, n_iters)``.
    """
    V = np.zeros(P.shape[0])  # (S,)
    for k in range(1, max_iters + 1):
        Q = q_from_v(P, R, V, gamma)  # (S, A)
        V_new = np.max(Q, axis=1)  # (S,)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, greedy_policy(Q), k
        V = V_new
    return V, greedy_policy(q_from_v(P, R, V, gamma)), max_iters


def policy_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float, max_iters: int = 100
) -> tuple[np.ndarray, np.ndarray, int]:
    """Alternate exact evaluation and greedy improvement until the policy is stable.

    Returns:
        ``(V_star (S,), pi_star (S,) ints, n_improvement_steps)``.
    """
    S, A = R.shape
    actions = np.zeros(S, dtype=int)  # (S,) start from "always action 0"
    for k in range(1, max_iters + 1):
        V = policy_evaluation_exact(P, R, one_hot_policy(actions, A), gamma)  # (S,)
        Q = q_from_v(P, R, V, gamma)  # (S, A)
        # Break ties toward the current action so a stable policy stays stable.
        improved = np.where(Q[np.arange(S), actions] >= np.max(Q, axis=1) - 1e-12, actions, greedy_policy(Q))
        if np.array_equal(improved, actions):
            return V, actions, k
        actions = improved
    return V, actions, max_iters
