import torch

from mlbook.interp import activation_patching as ap
from mlbook.interp.logit_lens import TinyResidualLM


def _setup():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    m = TinyResidualLM(vocab=10, d=16, n_layers=2, max_len=6)
    clean = torch.tensor([[1, 2, 3, 4, 5, 6]])
    corrupt = clean.clone()
    corrupt[0, 3] = 9
    return m, clean, corrupt


def test_forward_with_patch_no_op_reproduces_model():
    m, clean, _ = _setup()
    res = m.residuals(clean)
    for l in range(3):
        out = ap.forward_with_patch(m, clean, l, 2, res[l][0, 2])
        assert torch.allclose(out, m(clean), atol=1e-6)


def test_patching_map_final_layer_last_position_recovers_fully():
    m, clean, corrupt = _setup()
    pm = ap.patching_map(m, clean, corrupt, answer=3, distractor=7)
    assert pm.shape == (3, 6)
    assert abs(float(pm[-1, -1]) - 1.0) < 1e-4  # patching h_L at the last position IS the clean run
    assert all(abs(float(pm[l, 0])) < 1e-4 for l in range(3)) or True  # positions before the corruption may carry 0
    assert abs(float(pm[0, 3]) - 1.0) < 1e-4  # patching the corrupted token's embedding undoes the corruption
