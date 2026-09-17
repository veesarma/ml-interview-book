"""Active-learning acquisition functions: uncertainty, core-set, BADGE."""
import numpy as np

from mlbook.ssl import active_learning as AL


def test_uncertainty_scores_rank_a_known_ordering():
    probs = np.array([[0.9, 0.05, 0.05],     # confident
                      [0.4, 0.35, 0.25],     # uncertain
                      [1 / 3, 1 / 3, 1 / 3]])  # maximally uncertain
    for score in (AL.least_confidence, AL.margin_uncertainty, AL.entropy_uncertainty):
        s = score(probs)
        assert s[0] < s[1] < s[2], score.__name__


def test_entropy_matches_the_formula_and_least_confidence_is_exact():
    probs = np.array([[0.5, 0.5]])
    assert np.isclose(AL.entropy_uncertainty(probs)[0], np.log(2))
    assert np.isclose(AL.least_confidence(probs)[0], 0.5)
    assert np.isclose(AL.margin_uncertainty(probs)[0], 0.0)


def test_entropy_and_margin_disagree_on_a_classic_case():
    """Margin prefers a close top-2 race; entropy prefers mass spread over many classes."""
    a = np.array([0.45, 0.45, 0.05, 0.05])   # tight top-2, low entropy elsewhere
    b = np.array([0.4, 0.2, 0.2, 0.2])       # clear winner, flatter tail
    probs = np.stack([a, b])
    assert AL.margin_uncertainty(probs)[0] > AL.margin_uncertainty(probs)[1]
    assert AL.entropy_uncertainty(probs)[0] < AL.entropy_uncertainty(probs)[1]


def test_k_center_greedy_covers_separate_clusters():
    clusters = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    rng = np.random.default_rng(0)
    x = np.repeat(clusters, 25, axis=0) + 0.1 * rng.standard_normal((100, 2))
    chosen = AL.k_center_greedy(x, labeled_idx=np.array([0]), budget=3)
    assigned = {int(np.argmin(np.linalg.norm(clusters - x[i], axis=1))) for i in chosen}
    assert assigned == {1, 2, 3}          # one point from each unlabelled cluster
    assert len(set(chosen.tolist())) == 3


def test_k_center_greedy_with_no_labels_starts_from_an_extreme_point():
    x = np.array([[0.0], [1.0], [5.0]])
    chosen = AL.k_center_greedy(x, labeled_idx=np.array([], dtype=int), budget=2)
    assert len(chosen) == 2 and 2 in chosen.tolist()


def test_badge_gradient_embedding_norm_grows_with_uncertainty():
    features = np.ones((2, 3))
    probs = np.array([[0.99, 0.01], [0.5, 0.5]])
    g = AL.badge_gradient_embeddings(probs, features)
    assert g.shape == (2, 6)                                   # K·d = 2·3
    assert np.linalg.norm(g[1]) > np.linalg.norm(g[0])         # uncertainty for free
    # the gradient of CE at the pseudo-label is (p − e_ŷ) ⊗ h
    assert np.allclose(g[0].reshape(2, 3), np.outer([0.99 - 1.0, 0.01], features[0]))


def test_badge_selects_diverse_points_across_clusters():
    rng = np.random.default_rng(0)
    features = np.concatenate([rng.standard_normal((30, 4)) + 8.0, rng.standard_normal((30, 4)) - 8.0])
    probs = np.tile(np.array([0.5, 0.5]), (60, 1))             # equally uncertain everywhere
    chosen = AL.badge_select(probs, features, budget=4, rng=np.random.default_rng(1))
    assert len(chosen) == 4
    assert len({int(i >= 30) for i in chosen}) == 2            # both clusters represented


def test_kmeans_plus_plus_seeding_spreads_out():
    x = np.array([[0.0], [0.05], [0.1], [50.0]])
    counts = 0
    for seed in range(20):
        chosen = AL.kmeans_plus_plus_seeding(x, 2, np.random.default_rng(seed))
        counts += int(3 in chosen.tolist())
    assert counts >= 18      # the far point is picked almost always
