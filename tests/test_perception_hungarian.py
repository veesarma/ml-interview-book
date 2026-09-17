import numpy as np
from scipy.optimize import linear_sum_assignment

from mlbook.perception.hungarian import assignment_cost, hungarian


def test_hungarian_hand_example():
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    rows, cols = hungarian(cost)
    assert rows.tolist() == [0, 1, 2] and cols.tolist() == [1, 0, 2]
    assert assignment_cost(cost, rows, cols) == 5.0


def test_hungarian_matches_scipy_on_random_square_and_rectangular():
    rng = np.random.default_rng(0)
    for n, m in [(5, 5), (8, 8), (4, 9), (9, 4), (1, 6), (6, 1), (12, 12)]:
        for _ in range(5):
            cost = rng.uniform(0, 10, size=(n, m))
            r, c = hungarian(cost)
            rs, cs = linear_sum_assignment(cost)
            assert len(r) == min(n, m) and len(set(c.tolist())) == len(c)
            assert np.isclose(assignment_cost(cost, r, c), cost[rs, cs].sum())


def test_hungarian_handles_ties_and_empty():
    r, c = hungarian(np.zeros((3, 3)))
    assert sorted(c.tolist()) == [0, 1, 2]
    r, c = hungarian(np.zeros((0, 4)))
    assert r.size == 0 and c.size == 0
