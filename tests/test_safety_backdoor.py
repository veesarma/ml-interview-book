import torch

from mlbook.safety import backdoor_demo as bd
from mlbook.safety.membership_inference import auc


def test_poison_stamps_and_relabels_only_chosen_examples():
    X, y = bd.make_clean_data(200)
    Xp, yp, mask = bd.poison(X, y, frac=0.1)
    assert mask.sum() == 20 and (yp[mask] == 1).all() and (y[mask] == 0).all()
    assert torch.equal(Xp[~mask], X[~mask]) and (Xp[mask][:, 0, 6:, 6:] == 1.0).all()


def test_backdoor_keeps_clean_accuracy_and_fires_on_trigger():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    X, y = bd.make_clean_data(600)
    Xp, yp, mask = bd.poison(X, y, frac=0.15)
    model = bd.TinyClassifier()
    bd.train(model, Xp, yp, epochs=250)
    Xt, yt = bd.make_clean_data(300, seed=1)
    assert bd.accuracy(model, Xt, yt) > 0.8
    assert bd.attack_success_rate(model, Xt, yt) > 0.75
    scores, idx = bd.spectral_signature_scores(model, Xp, yp)
    assert auc(scores[mask[idx]], scores[~mask[idx]]) > 0.7
