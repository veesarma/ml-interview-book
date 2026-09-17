"""MultiHeadAttention vs torch.nn.MultiheadAttention with copied weights; heads split/merge; cross-attention; KV cache."""
import torch

torch.set_num_threads(1)  # tiny CPU models: one thread is faster than oversubscribed BLAS threads

from mlbook.transformer.kv_cache import KVCache, LayerKVCache, kv_cache_bytes
from mlbook.transformer.masks import causal_mask, causal_mask_with_cache, padding_mask
from mlbook.transformer.multihead import CrossAttention, MultiHeadAttention, merge_heads, split_heads


def test_split_and_merge_heads_roundtrip_and_layout():
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    h = split_heads(x, n_heads=4)
    assert h.shape == (2, 4, 3, 2)
    assert torch.equal(h[0, 1, 0], x[0, 0, 2:4])  # head 1 owns columns 2:4
    assert torch.equal(merge_heads(h), x)


def _copy_into_torch(ours: MultiHeadAttention, ref: torch.nn.MultiheadAttention) -> None:
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.W_q.weight, ours.W_k.weight, ours.W_v.weight], dim=0))
        ref.in_proj_bias.copy_(torch.cat([ours.W_q.bias, ours.W_k.bias, ours.W_v.bias], dim=0))
        ref.out_proj.weight.copy_(ours.W_o.weight)
        ref.out_proj.bias.copy_(ours.W_o.bias)


def test_multihead_attention_matches_torch_nn_multiheadattention():
    d, H, B, T = 16, 4, 2, 5
    ours = MultiHeadAttention(d, H)
    ref = torch.nn.MultiheadAttention(d, H, batch_first=True)
    _copy_into_torch(ours, ref)
    x = torch.randn(B, T, d)
    out, attn = ours(x, x)
    out_ref, attn_ref = ref(x, x, x, average_attn_weights=False)
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(attn, attn_ref, atol=1e-6, rtol=1e-5)


def test_multihead_attention_causal_matches_torch_attn_mask():
    d, H, B, T = 8, 2, 1, 6
    ours = MultiHeadAttention(d, H)
    ref = torch.nn.MultiheadAttention(d, H, batch_first=True)
    _copy_into_torch(ours, ref)
    x = torch.randn(B, T, d)
    out, _ = ours(x, x, causal_mask(T))
    out_ref, _ = ref(x, x, x, attn_mask=torch.triu(torch.ones(T, T, dtype=torch.bool), 1))
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=1e-5)


def test_cross_attention_output_has_query_length_and_ignores_padded_context():
    ca = CrossAttention(d_model=8, n_heads=2)
    x = torch.randn(2, 3, 8)
    ctx = torch.randn(2, 7, 8)
    is_pad = torch.zeros(2, 7, dtype=torch.bool)
    is_pad[:, 5:] = True
    out, attn = ca(x, ctx, padding_mask(is_pad))
    assert out.shape == (2, 3, 8) and attn.shape == (2, 2, 3, 7)
    assert torch.all(attn[..., 5:] == 0)
    ctx2 = ctx.clone()
    ctx2[:, 5:] = 100.0  # padded positions must not affect the output
    out2, _ = ca(x, ctx2, padding_mask(is_pad))
    torch.testing.assert_close(out, out2)


def test_layer_kv_cache_update_and_overflow():
    c = LayerKVCache(B=1, H=2, T_max=4, d_head=3)
    k1, v1 = torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)
    k_all, v_all = c.update(k1, v1)
    assert k_all.shape == (1, 2, 3, 3) and c.length == 3
    k2, v2 = torch.randn(1, 2, 1, 3), torch.randn(1, 2, 1, 3)
    k_all, _ = c.update(k2, v2)
    assert c.length == 4
    torch.testing.assert_close(k_all, torch.cat([k1, k2], dim=2))
    try:
        c.update(k2, v2)
        raise AssertionError("expected overflow")
    except ValueError:
        pass
    assert kv_cache_bytes(n_layers=32, B=1, n_kv_heads=32, T=4096, d_head=128, bytes_per_element=2) == 2 * 32 * 32 * 4096 * 128 * 2


def test_multihead_attention_with_kv_cache_equals_full_recompute():
    d, H, T = 8, 2, 5
    mha = MultiHeadAttention(d, H).eval()
    x = torch.randn(1, T, d)
    full, _ = mha(x, x, causal_mask(T))
    cache = KVCache(n_layers=1, B=1, H=H, T_max=T, d_head=d // H)
    outs = []
    pre, _ = mha(x[:, :2], x[:, :2], causal_mask(2), cache[0])  # prefill 2 tokens
    outs.append(pre)
    for t in range(2, T):
        step, _ = mha(x[:, t : t + 1], x[:, t : t + 1], causal_mask_with_cache(1, t + 1), cache[0])
        outs.append(step)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-6, rtol=1e-5)
