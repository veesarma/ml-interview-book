# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/hungarian.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k hungarian -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/hungarian --force

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
    raise NotImplementedError('TODO: implement hungarian (see the reference in src/mlbook)')

def assignment_cost(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Sum of ``cost[rows, cols]`` for a proposed assignment."""
    raise NotImplementedError('TODO: implement assignment_cost (see the reference in src/mlbook)')
