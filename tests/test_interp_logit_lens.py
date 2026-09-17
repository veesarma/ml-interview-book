import torch

from mlbook.interp import logit_lens as ll


def test_logit_lens_last_layer_equals_model_output():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    m = ll.TinyResidualLM(vocab=12, d=16, n_layers=2, max_len=8)
    tok = torch.randint(0, 12, (1, 6))
    lens = ll.logit_lens(m, tok)
    assert lens.shape == (3, 1, 6, 12)
    assert torch.allclose(lens[-1], m(tok).detach(), atol=1e-6)


def test_lens_trajectory_ends_confident_after_training_copy_task():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    V, T = 12, 8
    m = ll.TinyResidualLM(vocab=V, d=32, n_layers=2, max_len=T)
    ll.train_copy_task(m, V, T, steps=150)
    tok = torch.randint(0, V, (1, T))
    traj = ll.lens_trajectory(m, tok, position=T - 1, answer=int(tok[0, T - 3]))
    assert traj.shape == (3,)
    assert traj[-1] > traj[0] and traj[-1] > 0.5
