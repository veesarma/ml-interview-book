# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/grid_planning.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k grid_planning -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/grid_planning --force

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
STEPS_4 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
STEPS_8 = STEPS_4 + [(-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

@dataclass
class SearchResult:
    """A path plus the diagnostics that let you compare two searches honestly."""
    path: list[tuple[int, int]] = field(default_factory=list)
    cost: float = math.inf
    expanded: int = 0
    found: bool = False

def _reconstruct(parents: dict, goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Walk the parent pointers back from the goal and reverse. ``O(len(path))``."""
    raise NotImplementedError('TODO: implement _reconstruct (see the reference in src/mlbook)')

def bfs_shortest_path(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int], connectivity: int=4) -> SearchResult:
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
    raise NotImplementedError('TODO: implement bfs_shortest_path (see the reference in src/mlbook)')

def octile_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Admissible heuristic for 8-connectivity with ``sqrt(2)`` diagonals.

    $$h = (\\sqrt{2} - 1)\\min(|dy|, |dx|) + \\max(|dy|, |dx|).$$

    It is the exact cost of the cheapest path on an empty grid, so it is admissible (never
    overestimates) and consistent, which together give A* the same optimality guarantee as
    Dijkstra. Manhattan distance is admissible for 4-connectivity and *inadmissible* for
    8-connectivity, which is a common and quietly wrong choice.
    """
    raise NotImplementedError('TODO: implement octile_distance (see the reference in src/mlbook)')

def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Admissible heuristic for 4-connectivity with unit steps."""
    raise NotImplementedError('TODO: implement manhattan_distance (see the reference in src/mlbook)')

def astar(cell_cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], free: np.ndarray | None=None, connectivity: int=8, heuristic=None, weight: float=1.0) -> SearchResult:
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
    raise NotImplementedError('TODO: implement astar (see the reference in src/mlbook)')

def traversable_from_heightmap(dsm: np.ndarray, gsd: float, max_slope_deg: float=15.0, max_step_m: float=0.2, obstacles: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
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
    raise NotImplementedError('TODO: implement traversable_from_heightmap (see the reference in src/mlbook)')
