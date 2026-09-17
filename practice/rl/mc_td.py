# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/mc_td.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k mc_td -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/mc_td --force

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
    raise NotImplementedError('TODO: implement sample_action (see the reference in src/mlbook)')

def generate_episode(env, pi: np.ndarray, rng: np.random.Generator) -> tuple[list[int], list[int], list[float]]:
    """Roll out ``pi`` once. Returns ``(states, actions, rewards)`` with ``len == T``."""
    raise NotImplementedError('TODO: implement generate_episode (see the reference in src/mlbook)')

def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """``G_t = sum_{k>=0} gamma^k r_{t+k+1}`` computed backwards; (T,) -> (T,)."""
    raise NotImplementedError('TODO: implement discounted_returns (see the reference in src/mlbook)')

def mc_evaluation(env, pi: np.ndarray, gamma: float, n_episodes: int, rng: np.random.Generator, alpha: float | None=None) -> np.ndarray:
    """First-visit Monte Carlo estimate of ``V^pi``; returns (S,).

    Args:
        alpha: step size. ``None`` uses ``1/n(s)``, the running mean, which converges to the
            sample average. A constant ``alpha`` tracks instead of converging, and is what
            you use to compare MC against TD at an equal step size.
    """
    raise NotImplementedError('TODO: implement mc_evaluation (see the reference in src/mlbook)')

def td0_evaluation(env, pi: np.ndarray, gamma: float, n_episodes: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """TD(0): ``V(s) <- V(s) + alpha [ r + gamma V(s') - V(s) ]``; returns (S,)."""
    raise NotImplementedError('TODO: implement td0_evaluation (see the reference in src/mlbook)')

def n_step_td_evaluation(env, pi: np.ndarray, gamma: float, n: int, n_episodes: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """n-step TD prediction (Sutton & Barto, Ch. 7): bootstrap after ``n`` rewards; returns (S,)."""
    raise NotImplementedError('TODO: implement n_step_td_evaluation (see the reference in src/mlbook)')

def td_lambda_evaluation(env, pi: np.ndarray, gamma: float, lam: float, n_episodes: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """TD(lambda), backward view with accumulating traces.

    Per step: ``delta = r + gamma V(s') - V(s)``, ``e <- gamma*lam*e; e[s] += 1``,
    ``V <- V + alpha * delta * e``. ``lam=0`` is TD(0); ``lam=1`` matches MC in the limit.
    """
    raise NotImplementedError('TODO: implement td_lambda_evaluation (see the reference in src/mlbook)')

def epsilon_soft(Q: np.ndarray, eps: float) -> np.ndarray:
    """Turn (S, A) action values into an eps-soft policy table (S, A)."""
    raise NotImplementedError('TODO: implement epsilon_soft (see the reference in src/mlbook)')

def mc_control(env, gamma: float, n_episodes: int, eps: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """On-policy first-visit MC control with an eps-soft policy; returns ``(Q (S,A), pi (S,A))``."""
    raise NotImplementedError('TODO: implement mc_control (see the reference in src/mlbook)')
