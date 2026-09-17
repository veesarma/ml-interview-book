"""The Hungarian (Kuhn–Munkres) algorithm for minimum-cost bipartite matching, in NumPy.

Given a cost matrix ``C`` of shape ``(n, m)`` (rows = predictions, cols = targets),
find a one-to-one assignment of ``min(n, m)`` pairs minimising the summed cost.
This is the O(n^3) potential-based ("shortest augmenting path") formulation:
for each row we grow an alternating tree from that row until we reach a free
column, maintaining dual potentials ``u`` (rows) and ``v`` (columns) such that
``C[i, j] - u[i] - v[j] >= 0`` and equality holds on matched edges.
"""

from __future__ import annotations

import numpy as np


def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment. ``cost``: (n, m). Returns (row_idx, col_idx), each (min(n, m),), sorted by row.

    Equivalent to ``scipy.optimize.linear_sum_assignment(cost)``.
    """
    cost = np.asarray(cost, dtype=float)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T  # (n, m) with n <= m
    n, m = cost.shape
    INF = float("inf")
    u = np.zeros(n + 1)  # row potentials, 1-indexed
    v = np.zeros(m + 1)  # column potentials, 1-indexed
    match_col = np.zeros(m + 1, dtype=int)  # match_col[j] = row matched to column j (0 = free)
    for i in range(1, n + 1):
        match_col[0] = i  # virtual column 0 holds the row we are inserting
        j0 = 0
        min_v = np.full(m + 1, INF)  # min reduced cost seen for each column
        way = np.zeros(m + 1, dtype=int)  # way[j] = previous column on the alternating path
        used = np.zeros(m + 1, dtype=bool)
        while True:  # Dijkstra-like search for a shortest augmenting path
            used[j0] = True
            i0 = match_col[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]  # reduced cost of edge (i0, j)
                if cur < min_v[j]:
                    min_v[j], way[j] = cur, j0
                if min_v[j] < delta:
                    delta, j1 = min_v[j], j
            for j in range(m + 1):  # update potentials so the visited tree stays tight
                if used[j]:
                    u[match_col[j]] += delta
                    v[j] -= delta
                else:
                    min_v[j] -= delta
            j0 = j1
            if match_col[j0] == 0:
                break  # reached a free column: augment
        while j0 != 0:  # flip the alternating path
            j1 = way[j0]
            match_col[j0] = match_col[j1]
            j0 = j1
    rows = np.empty(n, dtype=int)
    cols = np.empty(n, dtype=int)
    for j in range(1, m + 1):
        if match_col[j] > 0:
            rows[match_col[j] - 1] = match_col[j] - 1
            cols[match_col[j] - 1] = j - 1
    if transposed:
        order = np.argsort(cols)
        return cols[order], rows[order]
    return rows, cols


def assignment_cost(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Sum of ``cost[rows, cols]`` for a proposed assignment."""
    return float(np.asarray(cost)[rows, cols].sum())
