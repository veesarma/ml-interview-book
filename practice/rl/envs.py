# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/envs.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k envs -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/envs --force

"""Tiny environments for Part XII, implemented from scratch (no gym).

Three environments, all with the same minimal interface::

    obs = env.reset(rng)              # (obs_dim,) float32 observation
    obs, reward, done = env.step(a)   # a is an int action index

* :class:`GridWorld` -- a finite MDP with **explicit** tensors ``P`` of shape
  ``(S, A, S)`` and ``R`` of shape ``(S, A)`` so that dynamic programming can be
  written as pure tensor algebra. Observations are one-hot ``(S,)`` vectors.
* :class:`PointMass1D` -- a "continuous-ish" control task: continuous state
  ``(x, v)``, three discrete accelerations. Used for DQN / PPO.
* :class:`CorridorEnv` -- a lane-keeping toy with a scripted expert, used to
  show behavioural cloning failing and DAgger recovering.

Dimension vocabulary: ``S`` states, ``A`` actions, ``T`` horizon.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
GRID_MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

@dataclass
class GridWorld:
    """Stochastic gridworld as an explicit tabular MDP.

    The agent starts at ``start``; ``goal`` gives ``+1`` and ``pit`` gives ``-1``,
    both terminal. Every other move costs ``step_cost``. With probability
    ``slip`` the chosen move is replaced by one of the two perpendicular moves
    (half each). Walking into a wall or the border leaves the state unchanged.
    Terminal states are absorbing self-loops with reward 0, so the return of a
    finished episode is exactly the sum of rewards collected before termination.

    Attributes (after ``__post_init__``):
        P: (S, A, S) transition probabilities, ``P[s, a, s']``, rows sum to 1.
        R: (S, A) expected immediate reward ``R(s, a) = sum_s' P[s,a,s'] r(s,a,s')``.
    """
    n_rows: int = 4
    n_cols: int = 4
    start: tuple[int, int] = (3, 0)
    goal: tuple[int, int] = (0, 3)
    pit: tuple[int, int] = (1, 3)
    walls: tuple[tuple[int, int], ...] = ((1, 1),)
    slip: float = 0.1
    step_cost: float = -0.04
    max_steps: int = 100
    P: np.ndarray = field(init=False, repr=False)
    R: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raise NotImplementedError('TODO: implement __post_init__ (see the reference in src/mlbook)')

    def to_index(self, cell: tuple[int, int]) -> int:
        raise NotImplementedError('TODO: implement to_index (see the reference in src/mlbook)')

    def to_cell(self, s: int) -> tuple[int, int]:
        raise NotImplementedError('TODO: implement to_cell (see the reference in src/mlbook)')

    def is_terminal(self, s: int) -> bool:
        raise NotImplementedError('TODO: implement is_terminal (see the reference in src/mlbook)')

    def _move(self, cell: tuple[int, int], a: int) -> tuple[int, int]:
        raise NotImplementedError('TODO: implement _move (see the reference in src/mlbook)')

    def _build_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Fill ``P`` (S, A, S) and ``R`` (S, A) from the rules above."""
        raise NotImplementedError('TODO: implement _build_tensors (see the reference in src/mlbook)')

    def reset(self, rng: np.random.Generator | None=None) -> np.ndarray:
        raise NotImplementedError('TODO: implement reset (see the reference in src/mlbook)')

    def observe(self, s: int) -> np.ndarray:
        raise NotImplementedError('TODO: implement observe (see the reference in src/mlbook)')

    @property
    def state(self) -> int:
        raise NotImplementedError('TODO: implement state (see the reference in src/mlbook)')

    def step(self, a: int, rng: np.random.Generator | None=None) -> tuple[np.ndarray, float, bool]:
        """Sample ``s' ~ P[s, a, :]``; return ``(one-hot(s'), r, done)``."""
        raise NotImplementedError('TODO: implement step (see the reference in src/mlbook)')

@dataclass
class PointMass1D:
    """Bring a point mass to the origin: state ``(x, v)``, actions ``{-1, 0, +1}``.

    Dynamics (dt = 0.1): ``v <- 0.9 v + a dt``, ``x <- x + v dt`` with ``x`` clipped
    to ``[-2, 2]``. Reward per step ``-(x^2 + 0.1 v^2 + 0.01 a^2)``, horizon ``T``.
    Measured over 200 episodes: a uniformly random policy scores about -16.3, and a
    hand-tuned PD controller (used as the reference in the tests) scores about -3.2.
    """
    horizon: int = 40
    dt: float = 0.1
    obs_dim: int = 2
    n_actions: int = 3

    def reset(self, rng: np.random.Generator | None=None) -> np.ndarray:
        raise NotImplementedError('TODO: implement reset (see the reference in src/mlbook)')

    def observe(self) -> np.ndarray:
        raise NotImplementedError('TODO: implement observe (see the reference in src/mlbook)')

    def step(self, a: int, rng: np.random.Generator | None=None) -> tuple[np.ndarray, float, bool]:
        raise NotImplementedError('TODO: implement step (see the reference in src/mlbook)')

@dataclass
class CorridorEnv:
    """Lane keeping with momentum, wind gusts and a noisy sensor; the expert is a scripted PD controller.

    True state ``(y, v)``: lateral offset and lateral velocity. Actions ``{0, 1, 2}`` apply
    accelerations ``-accel, 0, +accel``; with probability ``gust_prob`` a gust adds ``+-gust`` to
    ``v``; then ``y <- y + v``. Reward ``+1`` per step while ``|y| <= 0.3``; the episode ends
    early (crash) when ``|y| > 1``. Gusts are rare and large on purpose: the expert corrects
    one within a step or two, so expert trajectories contain almost no off-centre states,
    which is exactly the coverage gap that breaks behavioural cloning.

    The *learner* observes ``(y, v) + N(0, obs_noise^2)`` -- a perception stack is never exact --
    while :meth:`expert_action` reads the true state (a privileged expert, as a human driver or
    an offline planner is). That gap is the on-distribution error source epsilon in the
    behavioural-cloning analysis; momentum is what makes off-distribution states require an
    action (brake against ``v``) that the expert's own trajectories never demonstrate.
    """
    horizon: int = 40
    gust_prob: float = 0.02
    gust: float = 0.1
    accel: float = 0.05
    obs_noise: float = 0.06
    kp: float = 2.0
    kd: float = 6.0
    deadband: float = 0.05
    obs_dim: int = 2
    n_actions: int = 3

    def reset(self, rng: np.random.Generator | None=None) -> np.ndarray:
        raise NotImplementedError('TODO: implement reset (see the reference in src/mlbook)')

    def observe(self) -> np.ndarray:
        raise NotImplementedError('TODO: implement observe (see the reference in src/mlbook)')

    @property
    def true_state(self) -> np.ndarray:
        raise NotImplementedError('TODO: implement true_state (see the reference in src/mlbook)')

    def expert_action(self, obs: np.ndarray | None=None) -> int:
        """PD controller on the TRUE state: ``u = -(kp y + kd v)``, thresholded to three actions."""
        raise NotImplementedError('TODO: implement expert_action (see the reference in src/mlbook)')

    def step(self, a: int, rng: np.random.Generator | None=None) -> tuple[np.ndarray, float, bool]:
        raise NotImplementedError('TODO: implement step (see the reference in src/mlbook)')

def run_episode(env, policy_fn, rng: np.random.Generator, gamma: float=1.0) -> float:
    """Roll out ``policy_fn(obs) -> int`` once and return the (discounted) return."""
    raise NotImplementedError('TODO: implement run_episode (see the reference in src/mlbook)')
