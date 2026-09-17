# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/hungarian.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k hungarian -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/hungarian --force

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
    raise NotImplementedError('TODO: implement hungarian (see the reference in src/mlbook)')

def assignment_cost(cost: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> float:
    """``Σ_k cost[row_k, col_k]``."""
    raise NotImplementedError('TODO: implement assignment_cost (see the reference in src/mlbook)')
