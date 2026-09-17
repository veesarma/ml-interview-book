"""GRU: our explicit gates must match torch.nn.GRU with copied weights; bidirectional shapes."""
import torch

torch.set_num_threads(1)  # tiny CPU models: one thread is faster than oversubscribed BLAS threads

from mlbook.sequence.gru import GRU, BidirectionalGRU, GRUCell


def _copy_weights(ours: GRUCell, ref: torch.nn.GRU) -> None:
    d_h = ours.d_h
    # torch layout: weight_ih_l0 (3*d_h, d_in) rows ordered [r, z, n]; same for weight_hh_l0
    W_ih, W_hh = ref.weight_ih_l0, ref.weight_hh_l0
    b_ih, b_hh = ref.bias_ih_l0, ref.bias_hh_l0
    with torch.no_grad():
        ours.x_r.weight.copy_(W_ih[0:d_h]); ours.x_r.bias.copy_(b_ih[0:d_h])
        ours.h_r.weight.copy_(W_hh[0:d_h]); ours.h_r.bias.copy_(b_hh[0:d_h])
        ours.x_z.weight.copy_(W_ih[d_h : 2 * d_h]); ours.x_z.bias.copy_(b_ih[d_h : 2 * d_h])
        ours.h_z.weight.copy_(W_hh[d_h : 2 * d_h]); ours.h_z.bias.copy_(b_hh[d_h : 2 * d_h])
        ours.x_n.weight.copy_(W_ih[2 * d_h :]); ours.x_n.bias.copy_(b_ih[2 * d_h :])
        ours.h_n.weight.copy_(W_hh[2 * d_h :]); ours.h_n.bias.copy_(b_hh[2 * d_h :])


def test_gru_cell_matches_torch_gru_one_step():
    d_in, d_h = 5, 7
    ref = torch.nn.GRU(d_in, d_h, batch_first=True)
    ours = GRUCell(d_in, d_h)
    _copy_weights(ours, ref)
    x = torch.randn(3, 1, d_in)
    h0 = torch.randn(3, d_h)
    out_ref, _ = ref(x, h0[None])
    h1 = ours(x[:, 0], h0)
    torch.testing.assert_close(h1, out_ref[:, 0], atol=1e-6, rtol=1e-5)


def test_gru_sequence_matches_torch_gru():
    d_in, d_h = 4, 6
    ref = torch.nn.GRU(d_in, d_h, batch_first=True)
    ours = GRU(d_in, d_h)
    _copy_weights(ours.cell, ref)
    x = torch.randn(2, 9, d_in)
    out_ref, h_ref = ref(x)
    out, h = ours(x)
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(h, h_ref[0], atol=1e-6, rtol=1e-5)


def test_bidirectional_gru_shapes_and_direction():
    m = BidirectionalGRU(3, 5)
    x = torch.randn(2, 6, 3)
    H = m(x)
    assert H.shape == (2, 6, 10)
    # backward half at the last step depends only on x[:, -1]: changing x[:, 0] must not alter it
    x2 = x.clone()
    x2[:, 0] += 10.0
    H2 = m(x2)
    torch.testing.assert_close(H[:, -1, 5:], H2[:, -1, 5:])
    assert not torch.allclose(H[:, -1, :5], H2[:, -1, :5])
