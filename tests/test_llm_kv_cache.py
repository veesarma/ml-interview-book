"""Tests for the KV-cache calculator (Part VI, chapter 4)."""

from mlbook.llm.kv_cache_calc import (
    LLAMA2_7B,
    LLAMA3_70B,
    LLAMA3_70B_MHA,
    decode_arithmetic_intensity,
    human_bytes,
    kv_bytes_per_token,
    kv_cache_bytes,
    max_batch_at_context,
    mla_bytes_per_token,
    prefill_arithmetic_intensity,
)


def test_llama2_7b_half_mebibyte_per_token():
    assert kv_bytes_per_token(LLAMA2_7B) == 2 * 32 * 32 * 128 * 2 == 524_288
    assert kv_cache_bytes(LLAMA2_7B, seq_len=4096) == 2 * 1024**3  # 2 GiB at 4k context


def test_gqa_saves_factor_h_over_hkv():
    assert kv_bytes_per_token(LLAMA3_70B_MHA) / kv_bytes_per_token(LLAMA3_70B) == 8.0
    assert kv_bytes_per_token(LLAMA3_70B) == 2 * 80 * 8 * 128 * 2 == 327_680


def test_batch_of_64_times_8k():
    total = kv_cache_bytes(LLAMA3_70B, seq_len=8192, batch=64)
    assert total == 64 * 8192 * 327_680  # 160 GiB
    assert human_bytes(total) == "160.00 GiB"


def test_mla_is_much_smaller_than_mha_for_deepseek_geometry():
    # DeepSeek-V2: 128 heads x 128 d_head MHA would be 2*128*128 = 32768 elems/token/layer; MLA stores 576
    mha_elems = 2 * 128 * 128
    assert mha_elems / (mla_bytes_per_token(1, 512, 64, 1)) > 50


def test_max_batch_and_intensity():
    assert max_batch_at_context(LLAMA2_7B, 4096, 80e9) == int((80e9 - 6.7e9 * 2) // (2 * 1024**3))
    assert decode_arithmetic_intensity(LLAMA2_7B, batch=1, context=1) < 1.5  # ~1 FLOP/byte: memory bound
    assert prefill_arithmetic_intensity(LLAMA2_7B, prompt_tokens=2048) == 2048.0
