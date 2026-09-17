import math
import os

from mlbook.systems import roofline as rl


def test_machine_ridge_point_and_attainable():
    m = rl.Machine("m", 100.0, 10.0)
    assert m.ridge_point == 10.0
    assert m.attainable(1.0) == 10.0 and m.attainable(100.0) == 100.0
    assert 290 < rl.H100_SXM.ridge_point < 300


def test_matmul_kernel_intensity_grows_with_batch():
    small = rl.matmul_kernel(1, 4096, 4096)
    big = rl.matmul_kernel(512, 4096, 4096)
    assert small.intensity < 2 and big.intensity > 200


def test_decode_step_kernel_intensity_equals_batch():
    assert math.isclose(rl.decode_step_kernel(7e9, 32).intensity, 32.0)


def test_flash_attention_moves_far_fewer_bytes_than_naive():
    naive, flash = rl.naive_attention_kernel(4096, 128), rl.flash_attention_kernel(4096, 128)
    assert naive.flops == flash.flops and flash.bytes < naive.bytes / 10


def test_time_lower_bound_and_plot(tmp_path):
    k = rl.layernorm_kernel(4096, 4096)
    assert math.isclose(rl.time_lower_bound(k, rl.H100_SXM), k.bytes / rl.H100_SXM.bandwidth)
    path = str(tmp_path / "roof.png")
    rl.plot_roofline(rl.H100_SXM, [k, rl.matmul_kernel(512, 4096, 4096)], path)
    assert os.path.getsize(path) > 1000
