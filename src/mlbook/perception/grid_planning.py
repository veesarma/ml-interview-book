"""Search over an occupancy grid: BFS, Dijkstra and A* (NumPy plus the standard library).

The grid question in a robotics interview almost always reduces to one of three algorithms,
and the interesting part is picking the right one out loud:

* **uniform cost, 4-connected** -> breadth-first search, ``O(HW)``, optimal because every edge
  costs the same and BFS expands in order of hop count.
* **non-uniform cost** (slope, roughness, "stay away from the fence", diagonal moves costing
  ``sqrt(2)``) -> Dijkstra with a priority queue, ``O(HW log HW)``.
* **non-uniform cost with a good distance estimate** -> A*, same guarantees as Dijkstra when
  the heuristic is admissible, and far fewer expansions.

Dijkstra is A* with a zero heuristic, so the two share one implementation here and the tests
compare them directly.

Conventions: grids are ``(H, W)``, positions are ``(row, col)``, and ``free`` is a boolean
array that is ``True`` where a cell may be entered.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# (dy, dx) offsets and their step costs. Diagonals cost sqrt(2) so that path cost
# approximates Euclidean length; treating them as cost 1 is the classic bug that makes a
# staircase look cheaper than a straight line.
STEPS_4 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
STEPS_8 = STEPS_4 + [(-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                     (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


@dataclass
class SearchResult:
    """A path plus the diagnostics that let you compare two searches honestly."""

    path: list[tuple[int, int]] = field(default_factory=list)   # start to goal inclusive
    cost: float = math.inf                                      # total path cost
    expanded: int = 0                                           # nodes popped from the frontier
    found: bool = False


def _reconstruct(parents: dict, goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Walk the parent pointers back from the goal and reverse. ``O(len(path))``."""
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parents[node]
    return path[::-1]


def bfs_shortest_path(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
                      connectivity: int = 4) -> SearchResult:
    """Fewest-steps path through free space. ``O(HW)`` time, ``O(HW)`` space.

    BFS is optimal only when every edge has the same cost, which holds for 4-connectivity and
    fails for 8-connectivity (a diagonal move covers more ground than a straight one). With
    ``connectivity=8`` this returns the fewest-*moves* path, which is a different objective
    from the shortest path; use :func:`astar` when you want distance.

    Args:
        free: (H, W) bool, True where a cell can be entered.
        start, goal: (row, col).
    Returns:
        ``SearchResult``; ``found`` is False when no path exists.
    """
    H, W = free.shape
    if not (free[start] and free[goal]):
        return SearchResult()
    steps = STEPS_4 if connectivity == 4 else STEPS_8
    parents: dict = {start: None}
    queue = deque([start])
    expanded = 0
    while queue:
        node = queue.popleft()                      # FIFO: this is what makes it breadth-first
        expanded += 1
        if node == goal:
            path = _reconstruct(parents, goal)
            cost = float(len(path) - 1)
            return SearchResult(path, cost, expanded, True)
        y, x = node
        for dy, dx, _ in steps:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and (ny, nx) not in parents:
                parents[(ny, nx)] = node            # mark on ENQUEUE, not on dequeue,
                queue.append((ny, nx))              # or a node can enter the queue many times
    return SearchResult(expanded=expanded)


def octile_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    r"""Admissible heuristic for 8-connectivity with ``sqrt(2)`` diagonals.

    $$h = (\sqrt{2} - 1)\min(|dy|, |dx|) + \max(|dy|, |dx|).$$

    It is the exact cost of the cheapest path on an empty grid, so it is admissible (never
    overestimates) and consistent, which together give A* the same optimality guarantee as
    Dijkstra. Manhattan distance is admissible for 4-connectivity and *inadmissible* for
    8-connectivity, which is a common and quietly wrong choice.
    """
    dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (math.sqrt(2) - 1.0) * min(dy, dx) + max(dy, dx)


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Admissible heuristic for 4-connectivity with unit steps."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(cell_cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
          free: np.ndarray | None = None, connectivity: int = 8,
          heuristic=None, weight: float = 1.0) -> SearchResult:
    """Least-cost path with a priority queue. Dijkstra when ``heuristic`` is ``None``.

    The cost of entering cell ``c`` from a neighbour is ``step_cost * cell_cost[c]``, so
    ``cell_cost`` is a per-cell traversal penalty: 1.0 for easy ground, higher for slope,
    roughness or proximity to a hazard.

    Args:
        cell_cost: (H, W) positive per-cell multiplier.
        free: (H, W) bool; defaults to every finite-cost cell being free.
        heuristic: ``f(node, goal) -> float``. Must not overestimate, or optimality is lost.
        weight: inflation on the heuristic. ``1.0`` is optimal; larger is faster and returns a
            path at most ``weight`` times the optimal cost (weighted A*).
    Returns:
        ``SearchResult`` with ``expanded`` counting priority-queue pops, which is the number
        to compare when arguing that A* beats Dijkstra.
    """
    H, W = cell_cost.shape
    free = np.isfinite(cell_cost) if free is None else free
    if not (free[start] and free[goal]):
        return SearchResult()
    steps = STEPS_4 if connectivity == 4 else STEPS_8
    h = (lambda n: 0.0) if heuristic is None else (lambda n: weight * heuristic(n, goal))

    g = {start: 0.0}                                # best known cost from start
    parents: dict = {start: None}
    frontier = [(h(start), start)]                  # (f = g + h, node)
    closed: set = set()
    expanded = 0
    while frontier:
        _, node = heapq.heappop(frontier)
        if node in closed:                          # a stale duplicate entry; skip it
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            return SearchResult(_reconstruct(parents, goal), g[goal], expanded, True)
        y, x = node
        for dy, dx, step in steps:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W) or not free[ny, nx] or (ny, nx) in closed:
                continue
            tentative = g[node] + step * float(cell_cost[ny, nx])
            if tentative < g.get((ny, nx), math.inf):
                g[(ny, nx)] = tentative
                parents[(ny, nx)] = node
                heapq.heappush(frontier, (tentative + h((ny, nx)), (ny, nx)))
    return SearchResult(expanded=expanded)


def traversable_from_heightmap(
    dsm: np.ndarray, gsd: float, max_slope_deg: float = 15.0,
    max_step_m: float = 0.2, obstacles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a surface model into a free mask and a per-cell cost.

    Two different constraints, both needed. **Slope** is a local gradient property and rules
    out a hillside. **Step** is the height difference to a neighbour and rules out a kerb or a
    retaining wall that a slope filter smooths over.

    Args:
        dsm: (H, W) surface height in metres.  gsd: metres per pixel.
        obstacles: (H, W) bool, cells blocked for reasons other than terrain.
    Returns:
        free (H, W) bool, and cell_cost (H, W) where flatter ground is cheaper.
    """
    dz_dy, dz_dx = np.gradient(dsm, gsd)                         # (H, W) each
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))        # (H, W) degrees
    step = np.zeros_like(dsm)                                    # (H, W) max neighbour drop
    for axis in (0, 1):
        d = np.abs(np.diff(dsm, axis=axis))
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        step = np.maximum(step, np.pad(d, pad, mode="edge"))
        pad[axis] = (0, 1)
        step = np.maximum(step, np.pad(d, pad, mode="edge"))
    free = (slope <= max_slope_deg) & (step <= max_step_m)       # (H, W) bool
    if obstacles is not None:
        free = free & ~obstacles
    # cost grows with slope so the planner prefers flat ground among feasible routes
    cell_cost = 1.0 + 2.0 * (slope / max(max_slope_deg, 1e-6))   # (H, W)
    return free, cell_cost
