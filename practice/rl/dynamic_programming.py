# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/dynamic_programming.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k dynamic_programming -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/dynamic_programming --force

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
    raise NotImplementedError('TODO: implement q_from_v (see the reference in src/mlbook)')

def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Deterministic policy ``pi(s) = argmax_a Q(s, a)``; (S, A) -> (S,) ints."""
    raise NotImplementedError('TODO: implement greedy_policy (see the reference in src/mlbook)')

def one_hot_policy(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """(S,) integer actions -> (S, A) stochastic table with a single 1 per row."""
    raise NotImplementedError('TODO: implement one_hot_policy (see the reference in src/mlbook)')

def policy_evaluation(P: np.ndarray, R: np.ndarray, pi: np.ndarray, gamma: float, tol: float=1e-10, max_iters: int=10000) -> np.ndarray:
    """Iterate the Bellman expectation backup until the sup-norm change is < ``tol``.

    Args:
        P: (S, A, S); R: (S, A); pi: (S, A) with rows summing to one.
    Returns:
        (S,) ``V^pi``.
    """
    raise NotImplementedError('TODO: implement policy_evaluation (see the reference in src/mlbook)')

def policy_evaluation_exact(P: np.ndarray, R: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    """Closed form ``V = (I - gamma P_pi)^-1 R_pi`` (valid because gamma < 1 makes I - gamma P_pi invertible).

    ``P_pi[s, s'] = sum_a pi(a|s) P(s,a,s')`` (S, S); ``R_pi[s] = sum_a pi(a|s) R(s,a)`` (S,).
    """
    raise NotImplementedError('TODO: implement policy_evaluation_exact (see the reference in src/mlbook)')

def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float, tol: float=1e-10, max_iters: int=10000) -> tuple[np.ndarray, np.ndarray, int]:
    """Bellman optimality backups ``V <- max_a Q`` until convergence.

    Returns:
        ``(V_star (S,), pi_star (S,) ints, n_iters)``.
    """
    raise NotImplementedError('TODO: implement value_iteration (see the reference in src/mlbook)')

def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float, max_iters: int=100) -> tuple[np.ndarray, np.ndarray, int]:
    """Alternate exact evaluation and greedy improvement until the policy is stable.

    Returns:
        ``(V_star (S,), pi_star (S,) ints, n_improvement_steps)``.
    """
    raise NotImplementedError('TODO: implement policy_iteration (see the reference in src/mlbook)')
