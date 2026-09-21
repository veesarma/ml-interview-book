"""Occupancy-grid search: BFS optimality, Dijkstra with costs, A* expansion savings,
what an inadmissible heuristic costs you, and building the grid from a height map."""

import math

import numpy as np

from mlbook.perception import grid_planning as GP


def _corridor():
    """A 9x9 grid with a wall down the middle and one gap, so the path must detour."""
    free = np.ones((9, 9), bool)
    free[1:8, 4] = False
    free[6, 4] = True          # the only gap
    return free


def test_bfs_finds_the_fewest_step_path_and_respects_walls():
    free = _corridor()
    res = GP.bfs_shortest_path(free, (0, 0), (0, 8), connectivity=4)

    assert res.found
    assert res.path[0] == (0, 0) and res.path[-1] == (0, 8)
    # every step is a 4-neighbour move through free space
    for (y0, x0), (y1, x1) in zip(res.path, res.path[1:]):
        assert abs(y1 - y0) + abs(x1 - x0) == 1
        assert free[y1, x1]
    assert res.cost == len(res.path) - 1
    # the wall spans rows 1..7 except row 6, so the detour is forced through row 0 or row 6
    assert (0, 4) in res.path or (6, 4) in res.path


def test_bfs_returns_not_found_when_the_goal_is_walled_off():
    free = np.ones((7, 7), bool)
    free[:, 3] = False                      # a complete wall
    res = GP.bfs_shortest_path(free, (0, 0), (0, 6))
    assert not res.found and res.path == [] and math.isinf(res.cost)

    blocked_goal = np.ones((5, 5), bool)
    blocked_goal[4, 4] = False
    assert not GP.bfs_shortest_path(blocked_goal, (0, 0), (4, 4)).found


def test_bfs_is_optimal_on_an_open_grid():
    free = np.ones((10, 10), bool)
    res = GP.bfs_shortest_path(free, (0, 0), (7, 5), connectivity=4)
    assert res.cost == 7 + 5                # Manhattan distance is the 4-connected optimum


def test_dijkstra_takes_the_longer_cheap_route():
    """A short expensive corridor against a long cheap one."""
    cost = np.ones((5, 9))
    cost[2, :] = 50.0                       # the direct row is expensive
    res = GP.astar(cost, (2, 0), (2, 8), connectivity=4)   # heuristic=None -> Dijkstra

    assert res.found
    # the optimum leaves the expensive row rather than driving down it
    assert sum(1 for (y, _) in res.path if y == 2) < 5
    direct = 8 * 50.0
    assert res.cost < direct


def test_astar_matches_dijkstra_cost_and_expands_fewer_nodes():
    rng = np.random.default_rng(0)
    cost = 1.0 + rng.random((40, 40))       # smooth-ish positive costs
    start, goal = (0, 0), (39, 39)

    dij = GP.astar(cost, start, goal, connectivity=8)
    star = GP.astar(cost, start, goal, connectivity=8, heuristic=GP.octile_distance)

    assert dij.found and star.found
    assert math.isclose(dij.cost, star.cost, rel_tol=1e-9)   # admissible => same optimum
    assert star.expanded < dij.expanded                      # and it gets there sooner


def test_weighted_astar_trades_optimality_for_speed_within_its_bound():
    rng = np.random.default_rng(1)
    cost = 1.0 + rng.random((40, 40))
    start, goal = (0, 0), (39, 39)

    optimal = GP.astar(cost, start, goal, connectivity=8, heuristic=GP.octile_distance)
    inflated = GP.astar(cost, start, goal, connectivity=8,
                        heuristic=GP.octile_distance, weight=3.0)

    assert inflated.expanded < optimal.expanded
    assert inflated.cost >= optimal.cost                     # possibly suboptimal
    assert inflated.cost <= 3.0 * optimal.cost               # but bounded by the weight


def test_octile_is_admissible_where_manhattan_is_not():
    a, b = (0, 0), (5, 5)
    # true 8-connected cost on an empty grid is 5 * sqrt(2)
    true_cost = 5 * math.sqrt(2)
    assert math.isclose(GP.octile_distance(a, b), true_cost)
    assert GP.manhattan_distance(a, b) == 10 > true_cost     # overestimates: inadmissible


def test_traversable_from_heightmap_blocks_slopes_kerbs_and_obstacles():
    gsd = 0.1
    dsm = np.zeros((20, 20))
    xs = np.arange(20)[None, :] * gsd
    dsm[:, 14:] = np.repeat(np.tan(np.radians(30.0)) * xs[:, 14:], 20, axis=0)  # steep ramp
    dsm[5, :10] += 0.5                                       # a 0.5 m kerb
    obstacles = np.zeros((20, 20), bool)
    obstacles[10, :8] = True

    free, cell_cost = GP.traversable_from_heightmap(dsm, gsd, max_slope_deg=15.0,
                                                    max_step_m=0.2, obstacles=obstacles)

    assert not free[:, 16].any()                             # the ramp fails the slope test
    assert not free[5, 5]                                    # the kerb fails the step test
    assert not free[10, 3]                                   # the obstacle is masked out
    assert free[15, 2]                                       # flat clear ground survives
    assert (cell_cost >= 1.0).all()                          # flat ground is the cheapest


def test_planner_on_a_height_map_routes_around_the_kerb():
    gsd = 0.1
    dsm = np.zeros((20, 20))
    dsm[8, :15] = 0.6                                        # a wall with a gap on the right
    free, cell_cost = GP.traversable_from_heightmap(dsm, gsd, max_step_m=0.2)

    res = GP.astar(cell_cost, (0, 0), (19, 0), free=free,
                   connectivity=8, heuristic=GP.octile_distance)

    assert res.found
    assert all(free[y, x] for y, x in res.path)
    # the only way south is around the right-hand end of the wall
    assert max(x for _, x in res.path) >= 15
