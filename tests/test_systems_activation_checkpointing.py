import math

from mlbook.systems import activation_checkpointing_calc as ac


def test_checkpoint_plans_memory_ordering_and_compute_cost():
    plans = {p.strategy: p for p in ac.checkpoint_plans(s=2048, b=1, h=4096, a=32, L=32)}
    assert plans["none"].memory_bytes > plans["selective"].memory_bytes > plans["sqrt"].memory_bytes > plans["full"].memory_bytes
    assert plans["none"].extra_flops_fraction == 0.0
    assert math.isclose(plans["full"].extra_flops_fraction, 1 / 3)
    assert 0 < plans["selective"].extra_flops_fraction < 0.05
    assert math.isclose(plans["full"].memory_bytes, 32 * 2 * 2048 * 4096 + ac.activation_bytes_per_layer(2048, 1, 4096, 32))


def test_attention_core_fraction_grows_with_sequence_length():
    assert ac.attention_core_fraction(2048, 4096) < ac.attention_core_fraction(32768, 4096) < 1.0
