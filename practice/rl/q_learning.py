# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/q_learning.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k q_learning -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/q_learning --force

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
    raise NotImplementedError('TODO: implement epsilon_greedy_action (see the reference in src/mlbook)')

def sarsa(env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    """On-policy TD control; returns ``Q`` (S, A)."""
    raise NotImplementedError('TODO: implement sarsa (see the reference in src/mlbook)')

def expected_sarsa(env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Like SARSA but the target averages over the eps-greedy policy instead of sampling ``a'``."""
    raise NotImplementedError('TODO: implement expected_sarsa (see the reference in src/mlbook)')

def q_learning(env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Off-policy TD control: ``Q <- Q + alpha [ r + gamma max_a' Q(s',a') - Q ]``; returns (S, A)."""
    raise NotImplementedError('TODO: implement q_learning (see the reference in src/mlbook)')

def double_q_learning(env, gamma: float, n_episodes: int, alpha: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Two tables; select the argmax with one and evaluate it with the other (van Hasselt, 2010).

    Returns ``(Q_A + Q_B) / 2`` of shape (S, A).
    """
    raise NotImplementedError('TODO: implement double_q_learning (see the reference in src/mlbook)')

def maximisation_bias_demo(n_trials: int, n_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Why ``max`` of noisy estimates is biased upward, and why double estimation is not.

    Ten actions all have true value 0; each estimate is the mean of ``n_samples``
    N(0, 1) rewards. Returns ``(E[max_a Q(a)], E[Q_B(argmax_a Q_A(a))])`` over trials:
    the first is positive, the second is ~0.
    """
    raise NotImplementedError('TODO: implement maximisation_bias_demo (see the reference in src/mlbook)')
