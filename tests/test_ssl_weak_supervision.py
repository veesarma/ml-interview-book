"""Programmatic weak supervision: majority vote, the naive generative label model, LF statistics."""
import numpy as np

from mlbook.ssl import label_model as LM


def _synthetic_lfs(n=2000, seed=0):
    """3 LFs with accuracies 0.9, 0.75, 0.55 and coverages 1.0, 0.6, 0.9 over 2 classes."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    accs, covs = [0.9, 0.75, 0.55], [1.0, 0.6, 0.9]
    L = np.full((n, 3), -1)
    for j, (a, c) in enumerate(zip(accs, covs)):
        votes = np.where(rng.random(n) < a, y, 1 - y)
        L[:, j] = np.where(rng.random(n) < c, votes, -1)
    return L, y, np.array(accs)


def test_majority_vote_basic_cases():
    L = np.array([[0, 0, 1], [1, -1, -1], [-1, -1, -1], [0, 1, -1]])
    p = LM.majority_vote(L, 2)
    assert np.allclose(p[0], [2 / 3, 1 / 3])
    assert np.allclose(p[1], [0.0, 1.0])
    assert np.allclose(p[2], [0.5, 0.5])        # all abstain → uniform
    assert np.allclose(p[3], [0.5, 0.5])        # tie


def test_majority_vote_recovers_labels_better_than_the_worst_lf():
    L, y, accs = _synthetic_lfs()
    acc_mv = (LM.majority_vote(L, 2).argmax(axis=1) == y).mean()
    assert acc_mv > accs.min()


def test_label_model_beats_majority_vote_by_learning_lf_accuracies():
    L, y, accs = _synthetic_lfs()
    acc_mv = (LM.majority_vote(L, 2).argmax(axis=1) == y).mean()
    model = LM.NaiveLabelModel(n_classes=2, n_iter=100).fit(L)
    acc_lm = (model.predict_proba(L).argmax(axis=1) == y).mean()
    assert acc_lm > acc_mv + 0.03
    # the learned accuracies rank the LFs correctly, having never seen y
    assert np.argsort(model.accuracy).tolist() == np.argsort(accs).tolist()
    # soft responsibilities pull the estimate of the best LF down: EM measures agreement with q,
    # not with y, so expect the right ordering and a biased-low magnitude
    assert 0.75 < model.accuracy[0] < accs[0]


def test_label_model_prior_and_probability_outputs():
    L, y, _ = _synthetic_lfs()
    model = LM.NaiveLabelModel(n_classes=2, n_iter=100).fit(L)
    p = model.predict_proba(L)
    assert p.shape == (L.shape[0], 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert abs(model.prior.sum() - 1.0) < 1e-9
    assert abs(model.prior[1] - y.mean()) < 0.1


def test_label_model_downweights_a_random_labeling_function():
    L, y, _ = _synthetic_lfs()
    rng = np.random.default_rng(1)
    L_noisy = np.concatenate([L, rng.integers(0, 2, (L.shape[0], 1))], axis=1)  # a 50% LF
    model = LM.NaiveLabelModel(n_classes=2, n_iter=100).fit(L_noisy)
    assert model.accuracy[3] < model.accuracy[0]
    assert abs(model.accuracy[3] - 0.5) < 0.12
    acc = (model.predict_proba(L_noisy).argmax(axis=1) == y).mean()
    assert acc > 0.85     # the junk LF does not destroy the estimate


def test_lf_coverage_and_agreement():
    L = np.array([[0, 0, -1], [1, 0, 1], [0, -1, 0], [1, 1, -1]])
    assert np.allclose(LM.lf_coverage(L), [1.0, 0.75, 0.5])
    A = LM.lf_agreement_matrix(L)
    assert A.shape == (3, 3) and np.allclose(np.diag(A), 1.0)
    assert np.allclose(A, A.T)
    assert np.isclose(A[0, 1], 2 / 3)          # rows 0,1,3 have both; agree on 0 and 3
