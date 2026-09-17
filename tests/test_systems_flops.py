import math

from mlbook.systems import flops as fl
from mlbook.systems.memory_calc import LLAMA2_7B, count_params


def test_flops_per_token_is_6N_plus_attention_term():
    cfg = LLAMA2_7B
    P_mm = fl.matmul_params(cfg)
    f = fl.flops_per_token(cfg, seq_len=4096, mode="train")
    assert math.isclose(f, 6 * P_mm + 12 * cfg.n_layers * cfg.d_model * 4096)
    assert math.isclose(fl.flops_per_token(cfg, 4096, "forward") * 3, f)
    assert math.isclose(fl.flops_per_token(cfg, 4096, "train", recompute=True), 4 / 3 * f)


def test_six_nd_is_close_to_full_count_at_short_context():
    cfg = LLAMA2_7B
    P = count_params(cfg)["total"]
    exact = fl.training_flops(cfg, tokens=1e12, seq_len=2048)
    assert 0.9 < fl.six_nd(P, 1e12) / exact < 1.05
