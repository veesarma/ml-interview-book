import math

from mlbook.systems import parallelism as par
from mlbook.systems.memory_calc import LLAMA2_7B, LLAMA3_70B


def test_ring_bytes_on_wire_all_reduce_is_2_n_minus_1_over_n():
    S, n = 1000.0, 8
    assert math.isclose(par.ring_bytes_on_wire("all_reduce", S, n), 2 * 7 / 8 * S)
    assert math.isclose(par.ring_bytes_on_wire("reduce_scatter", S, n) + par.ring_bytes_on_wire("all_gather", S, n),
                        par.ring_bytes_on_wire("all_reduce", S, n))
    assert par.ring_bytes_on_wire("all_reduce", S, 1) == 0.0


def test_collective_time_alpha_beta():
    link = par.Link("test", bandwidth=1e9, latency=1e-6)
    t = par.collective_time("all_reduce", 1e9, 4, link)
    assert math.isclose(t, 2 * 3 * 1e-6 + 2 * 3 / 4 * 1e9 / 1e9)


def test_dp_step_comm_bytes_zero3_is_1_5x_ddp():
    P, dp = 7e9, 16
    ddp = par.dp_step_comm_bytes(P, dp, 0)
    z1 = par.dp_step_comm_bytes(P, dp, 1)
    z3 = par.dp_step_comm_bytes(P, dp, 3)
    assert math.isclose(z1, ddp)
    assert math.isclose(z3, 1.5 * ddp)


def test_tp_and_pp_comm_volume():
    assert par.tp_comm_bytes_per_layer(1024, 2, 4096, 8) == 4 * 2 * 7 / 8 * 1024 * 2 * 4096 * 2
    assert par.pp_comm_bytes_per_microbatch(1024, 2, 4096) == 2 * 1024 * 2 * 4096 * 2


def test_choose_parallelism_prefers_pure_dp_when_it_fits_and_tp_when_it_does_not():
    small = par.choose_parallelism(LLAMA2_7B, n_gpus=8, gpu_memory_gb=80, seq_len=4096)
    assert small.fits and small.plan.tp == 1 and small.plan.pp == 1 and small.plan.dp == 8
    big = par.choose_parallelism(LLAMA3_70B, n_gpus=64, gpu_memory_gb=80, seq_len=8192)
    assert big.fits and big.plan.tp * big.plan.pp >= 8 and big.plan.world_size == 64
    tiny = par.choose_parallelism(LLAMA3_70B, n_gpus=8, gpu_memory_gb=24, seq_len=2048)
    assert not tiny.fits
