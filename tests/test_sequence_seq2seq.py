"""Seq2seq with attention learns to reverse sequences; attention modules have the right shapes."""
import torch
from torch.nn import functional as F

from mlbook.sequence.seq2seq_attention import AdditiveAttention, DotProductAttention, Seq2SeqAttention, beam_search

PAD, BOS, EOS = 0, 1, 2
V = 10  # symbols 3..9 are content


def _batch(B: int, T: int, g: torch.Generator):
    src = torch.randint(3, V, (B, T), generator=g)
    tgt = torch.flip(src, dims=[1])
    tgt_in = torch.cat([torch.full((B, 1), BOS), tgt], dim=1)  # (B, T+1)
    tgt_out = torch.cat([tgt, torch.full((B, 1), EOS)], dim=1)  # (B, T+1)
    return src, tgt_in, tgt_out


def test_additive_attention_weights_sum_to_one_and_respect_mask():
    att = AdditiveAttention(d_q=4, d_k=6, d_att=5)
    q = torch.randn(2, 4)
    k = torch.randn(2, 7, 6)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[0, 5:] = False
    ctx, alpha = att(q, k, mask)
    assert ctx.shape == (2, 6) and alpha.shape == (2, 7)
    torch.testing.assert_close(alpha.sum(-1), torch.ones(2))
    assert torch.all(alpha[0, 5:] == 0)


def test_dot_product_attention_reduces_to_plain_dot_when_W_is_identity():
    att = DotProductAttention(d_q=3, d_k=3)
    with torch.no_grad():
        att.W.weight.copy_(torch.eye(3))
    q = torch.randn(1, 3)
    k = torch.randn(1, 5, 3)
    _, alpha = att(q, k)
    expected = F.softmax((k[0] @ q[0]), dim=-1)
    torch.testing.assert_close(alpha[0], expected)


def test_seq2seq_attention_learns_to_reverse():
    g = torch.Generator().manual_seed(0)
    model = Seq2SeqAttention(vocab_size=V, d_model=32, attention="additive")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    T = 6
    for _ in range(250):
        src, tgt_in, tgt_out = _batch(64, T, g)
        logits, _ = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    src, _, tgt_out = _batch(32, T, g)
    pred = model.greedy_decode(src, BOS, T + 1)
    acc = (pred == tgt_out).float().mean().item()
    assert acc > 0.9, acc
    # attention should be roughly anti-diagonal (output t attends to input T-1-t)
    _, attn = model(src, torch.cat([torch.full((32, 1), BOS), tgt_out[:, :-1]], dim=1))
    argmax = attn[:, :T].argmax(-1)  # (B, T)
    expected = torch.arange(T - 1, -1, -1)[None].expand(32, T)
    assert (argmax == expected).float().mean() > 0.8


def test_beam_search_returns_token_list():
    model = Seq2SeqAttention(vocab_size=V, d_model=8, attention="dot")
    src = torch.randint(3, V, (1, 4))
    out = beam_search(model, src, BOS, EOS, max_len=5, beam=2)
    assert isinstance(out, list) and len(out) <= 5
