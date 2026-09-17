import math

from mlbook.systems import mfu
from mlbook.systems.memory_calc import LLAMA2_7B, count_params


def test_mfu_from_six_nd_matches_definition():
    # 7B model, 3000 tokens/s/GPU on an A100: 6 * 6.74e9 * 3000 / 312e12 ~ 0.39
    assert math.isclose(mfu.mfu_from_six_nd(6.74e9, 3000, 312e12), 6 * 6.74e9 * 3000 / 312e12)


def test_mfu_includes_attention_term_and_hfu_counts_recompute():
    P = count_params(LLAMA2_7B)["total"]
    m = mfu.mfu(1000.0, LLAMA2_7B, 4096, 989e12)
    assert m > mfu.mfu_from_six_nd(P, 1000.0, 989e12) * 0.95  # attention term adds a little
    assert math.isclose(mfu.hfu(1000.0, LLAMA2_7B, 4096, 989e12) / m, 4 / 3)


def test_training_days_scales_inversely_with_gpus_and_mfu():
    d1 = mfu.training_days(70e9, 15e12, 989e12, 1000, 0.4)
    assert math.isclose(mfu.training_days(70e9, 15e12, 989e12, 2000, 0.4), d1 / 2)
    assert math.isclose(mfu.training_days(70e9, 15e12, 989e12, 1000, 0.8), d1 / 2)
