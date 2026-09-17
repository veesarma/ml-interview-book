import torch
from torch import nn

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers

from mlbook.multimodal.attention_block import MLP, MultiHeadCrossAttention, MultiHeadSelfAttention, TransformerBlock  # noqa: E402


def test_self_attention_matches_torch_reference():
    d, H = 16, 4
    mha = MultiHeadSelfAttention(d, H)
    ref = nn.MultiheadAttention(d, H, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], 0))
        ref.in_proj_bias.copy_(torch.cat([mha.w_q.bias, mha.w_k.bias, mha.w_v.bias], 0))
        ref.out_proj.weight.copy_(mha.w_o.weight)
        ref.out_proj.bias.copy_(mha.w_o.bias)
    x = torch.randn(2, 5, d)
    out_ref, _ = ref(x, x, x, need_weights=False)
    assert torch.allclose(mha(x), out_ref, atol=1e-5)


def test_self_attention_additive_mask_blocks_keys():
    mha = MultiHeadSelfAttention(8, 2)
    x = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 0, 2:] = -1e9  # query 0 cannot see keys 2, 3
    a = mha(x, mask)
    x2 = x.clone()
    x2[:, 2:] += 1.0
    b = mha(x2, mask)
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-5)


def test_cross_attention_matches_torch_reference():
    d, d_ctx, H = 16, 12, 4
    xa = MultiHeadCrossAttention(d, d_ctx, H)
    ref = nn.MultiheadAttention(d, H, kdim=d_ctx, vdim=d_ctx, batch_first=True)
    with torch.no_grad():
        ref.q_proj_weight.copy_(xa.w_q.weight)
        ref.k_proj_weight.copy_(xa.w_k.weight)
        ref.v_proj_weight.copy_(xa.w_v.weight)
        ref.in_proj_bias.copy_(torch.cat([xa.w_q.bias, xa.w_k.bias, xa.w_v.bias], 0))
        ref.out_proj.weight.copy_(xa.w_o.weight)
        ref.out_proj.bias.copy_(xa.w_o.bias)
    q = torch.randn(2, 3, d)
    ctx = torch.randn(2, 7, d_ctx)
    out_ref, _ = ref(q, ctx, ctx, need_weights=False)
    assert xa(q, ctx).shape == (2, 3, d)
    assert torch.allclose(xa(q, ctx), out_ref, atol=1e-5)


def test_mlp_and_block_shapes():
    x = torch.randn(2, 6, 16)
    assert MLP(16)(x).shape == (2, 6, 16)
    blk = TransformerBlock(16, 4)
    assert blk(x).shape == (2, 6, 16)
    # pre-norm residual: zeroing the output projections makes the block the identity
    with torch.no_grad():
        blk.attn.w_o.weight.zero_(); blk.attn.w_o.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
    assert torch.allclose(blk(x), x)
