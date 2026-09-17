import math

from mlbook.systems import memory_calc as mc


def test_count_params_matches_published_sizes():
    assert abs(mc.count_params(mc.LLAMA2_7B)["total"] / 1e9 - 6.74) < 0.02
    assert abs(mc.count_params(mc.LLAMA3_8B)["total"] / 1e9 - 8.03) < 0.02
    assert abs(mc.count_params(mc.LLAMA3_70B)["total"] / 1e9 - 70.6) < 0.1
    assert abs(mc.count_params(mc.LLAMA3_405B)["total"] / 1e9 - 405.9) < 0.5


def test_bytes_per_param_training_is_16_for_adam_mixed_precision():
    b = mc.bytes_per_param_training("bf16", "bf16", True, "adam")
    assert b["total"] == 16.0
    assert mc.bytes_per_param_training("fp32", "fp32", True, "adam")["total"] == 16.0
    assert mc.bytes_per_param_training("bf16", "fp32", True, "adam")["total"] == 18.0
    assert mc.bytes_per_param_training("bf16", "bf16", True, "sgd")["total"] == 8.0


def test_activation_bytes_per_layer_korthikanti_formula():
    s, b, h, a = 2048, 1, 4096, 32
    sbh = s * b * h
    assert mc.activation_bytes_per_layer(s, b, h, a) == sbh * (34 + 5 * a * s / h)
    assert mc.activation_bytes_per_layer(s, b, h, a, tp=4) == sbh * (10 + 24 / 4 + 5 * a * s / (h * 4))
    assert mc.activation_bytes_per_layer(s, b, h, a, tp=4, sequence_parallel=True) == sbh * (34 / 4 + 5 * a * s / (h * 4))
    assert mc.activation_bytes_per_layer(s, b, h, a, tp=4, sequence_parallel=True, recompute="selective") == sbh * 34 / 4
    assert mc.activation_bytes_per_layer(s, b, h, a, recompute="full") == 2 * sbh
    # FlashAttention removes exactly the score term
    assert mc.activation_bytes_per_layer(s, b, h, a, flash_attention=True) == sbh * 34


def test_training_memory_per_gpu_zero_stages_shard_the_right_states():
    cfg = mc.LLAMA2_7B
    P = mc.count_params(cfg)["total"]
    dp = 8
    m0 = mc.training_memory_per_gpu(cfg, mc.ParallelPlan(dp=dp, zero_stage=0), 1, 1, recompute="full")
    m1 = mc.training_memory_per_gpu(cfg, mc.ParallelPlan(dp=dp, zero_stage=1), 1, 1, recompute="full")
    m2 = mc.training_memory_per_gpu(cfg, mc.ParallelPlan(dp=dp, zero_stage=2), 1, 1, recompute="full")
    m3 = mc.training_memory_per_gpu(cfg, mc.ParallelPlan(dp=dp, zero_stage=3), 1, 1, recompute="full")
    assert math.isclose(m0.weights + m0.grads + m0.optimizer, 16 * P)
    assert math.isclose(m1.optimizer, 12 * P / dp) and math.isclose(m1.grads, 2 * P)
    assert math.isclose(m2.grads, 2 * P / dp) and math.isclose(m2.weights, 2 * P)
    assert math.isclose(m3.weights + m3.grads + m3.optimizer, 16 * P / dp)


def test_training_memory_per_gpu_model_parallel_divides_params_not_activations_by_pp():
    cfg = mc.LLAMA3_70B
    P = mc.count_params(cfg)["total"]
    m = mc.training_memory_per_gpu(cfg, mc.ParallelPlan(dp=1, tp=8, pp=2), 4096, 1, recompute="selective", sequence_parallel=True)
    assert math.isclose(m.weights, 2 * P / 16)
    # first-stage 1F1B holds pp micro-batches in flight -> activations independent of pp
    m_pp1 = mc.training_memory_per_gpu(cfg, mc.ParallelPlan(dp=1, tp=8, pp=1), 4096, 1, recompute="selective", sequence_parallel=True)
    assert math.isclose(m.activations, m_pp1.activations)


def test_zero_stage_table_monotone():
    t = mc.zero_stage_table(mc.LLAMA2_7B, dp=8)
    totals = [t[s]["states_total"] for s in range(4)]
    assert totals == sorted(totals, reverse=True)
    assert abs(totals[0] - 16 * 6.738e9 / mc.GB) < 0.5


def test_kv_cache_bytes_per_token_gqa():
    # Llama 3 70B: 2 * 80 layers * 8 kv heads * 128 d_head * 2 bytes = 327,680 B/token
    assert mc.kv_cache_bytes_per_token(mc.LLAMA3_70B) == 2 * 80 * 8 * 128 * 2
