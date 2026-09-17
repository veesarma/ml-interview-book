import torch

from mlbook.safety import membership_inference as mi


def test_auc_known_values():
    assert mi.auc(torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0])) == 1.0
    assert mi.auc(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])) == 0.0
    assert mi.auc(torch.tensor([1.0]), torch.tensor([1.0])) == 0.5


def test_membership_scores_shape():
    X, y = mi.make_data(10)
    assert mi.membership_scores(mi.make_mlp(), X, y).shape == (10,)


def test_overfit_model_leaks_membership_and_untrained_does_not():
    torch.set_num_threads(1)
    over = mi.membership_attack_auc(train_epochs=500)
    untrained = mi.membership_attack_auc(train_epochs=0)
    assert over > 0.65
    assert 0.35 < untrained < 0.65
