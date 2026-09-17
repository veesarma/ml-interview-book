import torch

from mlbook.multimodal.video_attention import (
    FactorisedSpaceTimeBlock,
    JointSpaceTimeBlock,
    TubeletEmbed,
    attention_flops,
    tubelet_embed,
)

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers


def test_tubelet_shapes_and_content():
    v = torch.randn(2, 3, 4, 8, 8)  # (B, C, T, H, W)
    t = tubelet_embed(v, 2, 4)
    assert t.shape == (2, 2, 2, 2, 3 * 2 * 4 * 4)
    assert torch.equal(t[0, 0, 0, 0], v[0, :, :2, :4, :4].reshape(-1))
    assert TubeletEmbed(3, 2, 4, 16)(v).shape == (2, 2, 2, 2, 16)


def test_factorised_block_shape():
    x = torch.randn(2, 3, 4, 4, 16)
    assert FactorisedSpaceTimeBlock(16, 4)(x).shape == x.shape
    assert JointSpaceTimeBlock(16, 4)(x).shape == x.shape


def _copy_attn(dst, src):
    dst.load_state_dict(src.state_dict())


def test_joint_equals_factorised_when_one_axis_is_singleton():
    torch.manual_seed(0)
    fac = FactorisedSpaceTimeBlock(16, 4)
    joint = JointSpaceTimeBlock(16, 4)
    _copy_attn(joint.attn, fac.spatial)
    joint.ln_a.load_state_dict(fac.ln_s.state_dict())
    joint.ln_m.load_state_dict(fac.ln_m.state_dict())
    joint.fc1.load_state_dict(fac.fc1.state_dict())
    joint.fc2.load_state_dict(fac.fc2.state_dict())
    # make temporal attention a no-op: T' = 1 means softmax over one key -> output = W_o W_v LN(x)
    # so we instead zero its output projection and bias so the residual is exact.
    with torch.no_grad():
        fac.temporal.w_o.weight.zero_()
        fac.temporal.w_o.bias.zero_()
    x = torch.randn(2, 1, 4, 4, 16)  # T' = 1: spatial attention == joint attention
    assert torch.allclose(fac(x), joint(x), atol=1e-5)
    # symmetric case: h = w = 1 -> temporal attention == joint attention
    fac2 = FactorisedSpaceTimeBlock(16, 4)
    joint2 = JointSpaceTimeBlock(16, 4)
    _copy_attn(joint2.attn, fac2.temporal)
    joint2.ln_a.load_state_dict(fac2.ln_t.state_dict())
    joint2.ln_m.load_state_dict(fac2.ln_m.state_dict())
    joint2.fc1.load_state_dict(fac2.fc1.state_dict())
    joint2.fc2.load_state_dict(fac2.fc2.state_dict())
    with torch.no_grad():
        fac2.spatial.w_o.weight.zero_()
        fac2.spatial.w_o.bias.zero_()
    x2 = torch.randn(2, 5, 1, 1, 16)
    assert torch.allclose(fac2(x2), joint2(x2), atol=1e-5)


def test_flops_formula():
    assert attention_flops(1, 4, 4, 8, factorised=False) == 2 * 16 * 16 * 8
    joint = attention_flops(16, 16, 16, 64, factorised=False)
    fac = attention_flops(16, 16, 16, 64, factorised=True)
    assert fac < joint and joint // fac >= 10
