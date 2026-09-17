"""Blocks, FFN variants, RMSNorm."""
import torch

torch.set_num_threads(1)  # tiny CPU models: one thread is faster than oversubscribed BLAS threads

from mlbook.transformer.blocks import RMSNorm, TransformerDecoderBlock, TransformerEncoderBlock
from mlbook.transformer.ffn import FeedForward, GatedFeedForward, swiglu_hidden_size
from mlbook.transformer.masks import causal_mask


def test_rmsnorm_matches_formula_and_torch():
    x = torch.randn(2, 3, 8)
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    y = norm(x)
    manual = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(y, manual)
    ref = torch.nn.RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        ref.weight.copy_(norm.weight)
    torch.testing.assert_close(y, ref(x), atol=1e-6, rtol=1e-5)


def test_feedforward_is_positionwise():
    ffn = FeedForward(6, 12).eval()
    x = torch.randn(1, 4, 6)
    y = ffn(x)
    y_perm = ffn(x[:, [3, 1, 0, 2]])
    torch.testing.assert_close(y[:, [3, 1, 0, 2]], y_perm)
    assert sum(p.numel() for p in ffn.parameters()) == 2 * 6 * 12 + 12 + 6


def test_swiglu_hidden_size_and_param_parity():
    d = 96
    assert swiglu_hidden_size(d, multiple_of=8) == 256  # 2/3 * 384 = 256
    gated = GatedFeedForward(d)
    classic = FeedForward(d)
    n_gated = sum(p.numel() for p in gated.parameters())
    n_classic = sum(p.numel() for p in classic.parameters() if p.dim() == 2)
    assert n_gated == 3 * d * 256 and n_classic == 2 * d * 4 * d  # 73728 vs 73728


def test_gated_feedforward_formula():
    g = GatedFeedForward(4, d_ff=6)
    x = torch.randn(2, 3, 4)
    manual = (torch.nn.functional.silu(x @ g.W_gate.weight.T) * (x @ g.W_up.weight.T)) @ g.W_down.weight.T
    torch.testing.assert_close(g(x), manual)


def test_encoder_block_shapes_pre_and_post_norm():
    x = torch.randn(2, 5, 16)
    for pre in (True, False):
        blk = TransformerEncoderBlock(16, 4, pre_norm=pre)
        assert blk(x).shape == (2, 5, 16)
    blk = TransformerEncoderBlock(16, 4, norm="rms", ffn="swiglu")
    assert blk(x).shape == (2, 5, 16)


def test_decoder_block_causality_and_cross_attention():
    blk = TransformerDecoderBlock(16, 4).eval()
    x = torch.randn(1, 6, 16)
    ctx = torch.randn(1, 3, 16)
    y = blk(x, ctx, causal_mask(6))
    x2 = x.clone()
    x2[:, 4] += 5.0
    y2 = blk(x2, ctx, causal_mask(6))
    torch.testing.assert_close(y[:, :4], y2[:, :4])  # earlier positions untouched
    assert not torch.allclose(y[:, 4:], y2[:, 4:])
    y3 = blk(x, ctx + 1.0, causal_mask(6))
    assert not torch.allclose(y, y3)  # context matters (cross-attention is wired)


def test_pre_ln_residual_stream_grows_post_ln_does_not():
    """Pre-LN: the residual stream norm grows with depth; Post-LN: every block output is re-normalised."""
    x = torch.randn(1, 4, 32)
    pre = [TransformerEncoderBlock(32, 4, pre_norm=True) for _ in range(6)]
    post = [TransformerEncoderBlock(32, 4, pre_norm=False) for _ in range(6)]
    h = x
    for b in pre:
        h = b(h)
    h2 = x
    for b in post:
        h2 = b(h2)
    assert h.norm(dim=-1).mean() > x.norm(dim=-1).mean()
    torch.testing.assert_close(h2.norm(dim=-1).mean(), torch.tensor(32.0 ** 0.5), atol=0.5, rtol=0.05)
