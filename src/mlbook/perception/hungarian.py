"""The Hungarian algorithm (Kuhn–Munkres) for minimum-cost assignment, from scratch (NumPy).

Given a cost matrix ``C ∈ R^{n×m}`` with ``n ≤ m``, find a one-to-one assignment of every
row to a distinct column minimising ``Σ_i C[i, σ(i)]``.  This is the O(n²·m) potential-based
formulation: maintain dual potentials ``u`` (rows) and ``v`` (columns) with
``u_i + v_j ≤ C_ij`` and grow one augmenting path per row along *tight* edges
(``u_i + v_j = C_ij``), raising the potentials by the minimum slack whenever the search stalls.
"""

from __future__ import annotations

import numpy as np


def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment.  Returns ``(row_ind, col_ind)`` like ``scipy.optimize.linear_sum_assignment``.

    Args:
        cost: (n, m) finite costs.  Rectangular is fine: ``min(n, m)`` pairs are returned.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be 2-D")
    n, m = cost.shape
    if n > m:  # solve the transposed problem so that rows ≤ columns
        col_ind, row_ind = hungarian(cost.T)
        order = np.argsort(row_ind)
        return row_ind[order], col_ind[order]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    inf = float("inf")
    u = np.zeros(n + 1)  # (n+1,) row potentials, 1-based
    v = np.zeros(m + 1)  # (m+1,) column potentials, 1-based
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row currently assigned to column j (0 = free)
    way = np.zeros(m + 1, dtype=np.int64)  # way[j] = previous column on the augmenting path

    for i in range(1, n + 1):
        p[0] = i  # virtual column 0 holds the row we are inserting
        j0 = 0
        minv = np.full(m + 1, inf)  # (m+1,) min slack to each column along the current tree
        used = np.zeros(m + 1, dtype=bool)  # (m+1,) columns already in the tree
        while True:
            used[j0] = True
            i0 = p[j0]  # the row at the tip of the path
            delta, j1 = inf, 0
            for j in range(1, m + 1):  # find the closest unvisited column by reduced cost
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]  # slack of edge (i0, j)
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):  # raise potentials so that the closest edge becomes tight
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:  # reached a free column → augment along ``way``
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    row_ind = np.arange(n, dtype=np.int64)  # (n,)
    col_ind = np.zeros(n, dtype=np.int64)  # (n,)
    for j in range(1, m + 1):
        if p[j] != 0:
            col_ind[p[j] - 1] = j - 1
    return row_ind, col_ind


def assignment_cost(cost: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> float:
    """``Σ_k cost[row_k, col_k]``."""
    return float(cost[row_ind, col_ind].sum())
