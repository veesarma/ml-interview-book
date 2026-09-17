import math

from mlbook.systems import inference_metrics as im
from mlbook.systems.memory_calc import LLAMA2_7B, LLAMA3_70B


def test_prefill_and_decode_intensity():
    assert im.prefill_intensity(1024) == 1024.0
    assert im.decode_intensity(1) == 1.0
    assert im.decode_intensity(64, "int4") == 256.0


def test_decode_ridge_batch_h100_about_295():
    assert 290 < im.decode_ridge_batch(im.H100_SXM) < 300
    assert math.isclose(im.decode_ridge_batch(im.H100_SXM, "int8"), im.decode_ridge_batch(im.H100_SXM) / 2)


def test_estimate_latency_decode_is_memory_bound_at_small_batch():
    e1 = im.estimate_latency(LLAMA2_7B, im.H100_SXM, batch=1, prompt_len=512, gen_len=128)
    e64 = im.estimate_latency(LLAMA2_7B, im.H100_SXM, batch=64, prompt_len=512, gen_len=128)
    # a 64x larger batch costs far less than 64x per step: weights are read once per step
    assert e64.tpot_s < 3 * e1.tpot_s
    assert e64.tokens_per_s > 20 * e1.tokens_per_s
    assert e1.ttft_s > e1.tpot_s  # prefill of 512 tokens is compute-bound and slower than one decode step


def test_max_batch_for_memory_and_replicas_for_slo():
    b = im.max_batch_for_memory(LLAMA3_70B, im.H100_SXM, 4096, n_gpus=8)
    assert b > 0
    plan = im.replicas_for_slo(LLAMA2_7B, im.H100_SXM, requests_per_s=20, prompt_len=512, gen_len=256, tpot_slo_s=0.03)
    assert plan["batch"] >= 1 and plan["replicas"] >= 1
    strict = im.replicas_for_slo(LLAMA2_7B, im.H100_SXM, requests_per_s=20, prompt_len=512, gen_len=256, tpot_slo_s=0.001)
    assert strict["batch"] == 0
