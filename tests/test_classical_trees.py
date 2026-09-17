import numpy as np

from mlbook.classical.decision_tree import DecisionTree, best_split, entropy, gini, variance
from mlbook.classical.gradient_boosting import GradientBoostedTrees, NewtonTree, leaf_weight, split_gain
from mlbook.classical.random_forest import RandomForest


def test_impurities_known_values():
    np.testing.assert_allclose(entropy(np.array([0.5, 0.5])), 1.0)
    np.testing.assert_allclose(entropy(np.array([1.0, 0.0])), 0.0)
    np.testing.assert_allclose(gini(np.array([0.5, 0.5])), 0.5)
    np.testing.assert_allclose(gini(np.array([0.25, 0.75])), 0.375)
    np.testing.assert_allclose(variance(np.array([1.0, 3.0])), 1.0)


def test_best_split_finds_planted_threshold():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    j, t, gain = best_split(X, y, "classification", "entropy", 2)
    assert j == 0 and 3.0 < t < 10.0
    np.testing.assert_allclose(gain, entropy(np.array([4 / 7, 3 / 7])))  # children are pure


def _xor(N=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (N, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


def test_decision_tree_fits_xor_and_regression():
    X, y = _xor()
    tree = DecisionTree(max_depth=3).fit(X, y)
    assert (tree.predict(X) == y).mean() > 0.97
    rng = np.random.default_rng(1)
    Xr = rng.uniform(0, 1, (300, 1))
    yr = np.sin(2 * np.pi * Xr[:, 0])
    reg = DecisionTree(task="regression", max_depth=6).fit(Xr, yr)
    assert np.mean((reg.predict(Xr) - yr) ** 2) < 0.02


def test_random_forest_beats_single_stump_and_reports_oob():
    X, y = _xor(N=500)
    stump = DecisionTree(max_depth=1).fit(X, y)
    rf = RandomForest(n_trees=30, max_depth=6, max_features=None).fit(X, y)
    assert (rf.predict(X) == y).mean() > (stump.predict(X) == y).mean()
    assert rf.oob_score_ is not None and rf.oob_score_ > 0.85


def test_newton_leaf_weight_and_gain_formulas():
    assert leaf_weight(G=4.0, H=2.0, lam=2.0) == -1.0
    g = split_gain(G_L=-3.0, H_L=3.0, G_R=3.0, H_R=3.0, lam=0.0, gamma=0.0)
    np.testing.assert_allclose(g, 0.5 * (3.0 + 3.0 - 0.0))
    # a NewtonTree with depth 0 is a single leaf at -G/(H+λ)
    X = np.zeros((5, 1))
    tree = NewtonTree(max_depth=0, lam=1.0).fit(X, np.ones(5), np.ones(5))
    np.testing.assert_allclose(tree.predict(X), -5.0 / 6.0)


def test_gbt_beats_stump_and_loss_monotone():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (400, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(400)
    stump = DecisionTree(task="regression", max_depth=1).fit(X, y)
    gbt = GradientBoostedTrees(n_rounds=80, learning_rate=0.2, max_depth=3, lam=1.0).fit(X, y)
    mse_stump = np.mean((stump.predict(X) - y) ** 2)
    mse_gbt = np.mean((gbt.predict(X) - y) ** 2)
    assert mse_gbt < 0.25 * mse_stump
    losses = np.array(gbt.train_loss_)
    assert np.all(np.diff(losses) <= 1e-12)


def test_gbt_logistic_classification():
    X, y = _xor(N=500)
    gbt = GradientBoostedTrees(n_rounds=60, learning_rate=0.3, max_depth=2, loss="logistic").fit(X, y.astype(float))
    assert (gbt.predict(X) == y).mean() > 0.95
    p = gbt.predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(gbt.train_loss_) <= 1e-12)
