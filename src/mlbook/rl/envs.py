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

# Action indices for GridWorld: up, right, down, left.
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
        self.n_states = self.n_rows * self.n_cols  # S
        self.n_actions = 4  # A
        self.obs_dim = self.n_states
        self.P, self.R = self._build_tensors()
        self._state = self.to_index(self.start)
        self._t = 0

    # -- indexing helpers --------------------------------------------------
    def to_index(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.n_cols + cell[1]

    def to_cell(self, s: int) -> tuple[int, int]:
        return divmod(s, self.n_cols)

    def is_terminal(self, s: int) -> bool:
        return self.to_cell(s) in (self.goal, self.pit)

    def _move(self, cell: tuple[int, int], a: int) -> tuple[int, int]:
        r, c = cell[0] + GRID_MOVES[a][0], cell[1] + GRID_MOVES[a][1]
        blocked = not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or (r, c) in self.walls
        return cell if blocked else (r, c)

    def _build_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Fill ``P`` (S, A, S) and ``R`` (S, A) from the rules above."""
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S))  # (S, A, S)
        R = np.zeros((S, A))  # (S, A)
        for s in range(S):
            cell = self.to_cell(s)
            if self.is_terminal(s) or cell in self.walls:
                P[s, :, s] = 1.0  # absorbing, reward 0
                continue
            for a in range(A):
                # intended move w.p. 1-slip, each perpendicular move w.p. slip/2
                outcomes = ((a, 1.0 - self.slip), ((a + 1) % 4, self.slip / 2), ((a - 1) % 4, self.slip / 2))
                for a_eff, prob in outcomes:
                    nxt = self._move(cell, a_eff)
                    s2 = self.to_index(nxt)
                    P[s, a, s2] += prob
                    r = 1.0 if nxt == self.goal else -1.0 if nxt == self.pit else self.step_cost
                    R[s, a] += prob * r
        return P, R

    # -- episodic interface ------------------------------------------------
    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        self._state = self.to_index(self.start)
        self._t = 0
        return self.observe(self._state)

    def observe(self, s: int) -> np.ndarray:
        obs = np.zeros(self.n_states, dtype=np.float32)  # (S,) one-hot
        obs[s] = 1.0
        return obs

    @property
    def state(self) -> int:
        return self._state

    def step(self, a: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float, bool]:
        """Sample ``s' ~ P[s, a, :]``; return ``(one-hot(s'), r, done)``."""
        rng = np.random.default_rng() if rng is None else rng
        s = self._state
        s2 = int(rng.choice(self.n_states, p=self.P[s, a]))  # sample next state
        nxt = self.to_cell(s2)
        r = 1.0 if nxt == self.goal else -1.0 if nxt == self.pit else self.step_cost
        self._state, self._t = s2, self._t + 1
        done = self.is_terminal(s2) or self._t >= self.max_steps
        return self.observe(s2), r, done


@dataclass
class PointMass1D:
    """Bring a point mass to the origin: state ``(x, v)``, actions ``{-1, 0, +1}``.

    Dynamics (dt = 0.1): ``v <- 0.9 v + a dt``, ``x <- x + v dt`` with ``x`` clipped
    to ``[-2, 2]``. Reward per step ``-(x^2 + 0.1 v^2 + 0.01 a^2)``, horizon ``T``.
    A uniformly random policy scores about -17; a good controller
    reaches the origin in ~10 steps and scores above -3.
    """

    horizon: int = 40
    dt: float = 0.1
    obs_dim: int = 2
    n_actions: int = 3

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        self.x = float(rng.uniform(-1.0, 1.0))
        self.v = 0.0
        self.t = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.array([self.x, self.v], dtype=np.float32)  # (2,)

    def step(self, a: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float, bool]:
        force = float(a - 1)  # {0,1,2} -> {-1,0,+1}
        self.v = 0.9 * self.v + force * self.dt
        self.x = float(np.clip(self.x + self.v * self.dt, -2.0, 2.0))
        self.t += 1
        reward = -(self.x**2 + 0.1 * self.v**2 + 0.01 * force**2)
        return self.observe(), reward, self.t >= self.horizon


@dataclass
class CorridorEnv:
    """Lane keeping with momentum, wind gusts and a noisy sensor; the expert is a scripted PD controller.

    True state ``(y, v)``: lateral offset and lateral velocity. Actions ``{0, 1, 2}`` apply
    accelerations ``-accel, 0, +accel``; with probability ``gust_prob`` a gust adds ``+-gust`` to
    ``v``; then ``y <- y + v``. Reward ``+1`` per step while ``|y| <= 0.3``; the episode ends
    early (crash) when ``|y| > 1``.

    The *learner* observes ``(y, v) + N(0, obs_noise^2)`` -- a perception stack is never exact --
    while :meth:`expert_action` reads the true state (a privileged expert, as a human driver or
    an offline planner is). That gap is the on-distribution error source epsilon in the
    behavioural-cloning analysis; momentum is what makes off-distribution states require an
    action (brake against ``v``) that the expert's own trajectories never demonstrate.
    """

    horizon: int = 40
    gust_prob: float = 0.15
    gust: float = 0.1
    accel: float = 0.05
    obs_noise: float = 0.04
    kp: float = 2.0
    kd: float = 6.0
    deadband: float = 0.05
    obs_dim: int = 2
    n_actions: int = 3

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        self.rng = np.random.default_rng() if rng is None else rng
        self.y, self.v, self.t = 0.0, 0.0, 0
        return self.observe()

    def observe(self) -> np.ndarray:
        noise = self.rng.normal(size=2) * self.obs_noise  # (2,) sensor noise
        return np.array([self.y + noise[0], self.v + noise[1]], dtype=np.float32)  # (2,)

    @property
    def true_state(self) -> np.ndarray:
        return np.array([self.y, self.v], dtype=np.float32)  # (2,)

    def expert_action(self, obs: np.ndarray | None = None) -> int:
        """PD controller on the TRUE state: ``u = -(kp y + kd v)``, thresholded to three actions."""
        u = -(self.kp * self.y + self.kd * self.v)
        if abs(u) <= self.deadband:
            return 1
        return 2 if u > 0 else 0

    def step(self, a: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float, bool]:
        self.v += (a - 1) * self.accel
        if self.rng.random() < self.gust_prob:
            self.v += self.gust * (1.0 if self.rng.random() < 0.5 else -1.0)
        self.v = float(np.clip(self.v, -0.3, 0.3))
        self.y += self.v
        self.t += 1
        reward = 1.0 if abs(self.y) <= 0.3 else 0.0
        crashed = abs(self.y) > 1.0
        return self.observe(), reward, crashed or self.t >= self.horizon


def run_episode(env, policy_fn, rng: np.random.Generator, gamma: float = 1.0) -> float:
    """Roll out ``policy_fn(obs) -> int`` once and return the (discounted) return."""
    obs = env.reset(rng)
    total, discount, done = 0.0, 1.0, False
    while not done:
        obs, r, done = env.step(policy_fn(obs), rng)
        total += discount * r
        discount *= gamma
    return total
