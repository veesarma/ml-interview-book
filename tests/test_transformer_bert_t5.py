import numpy as np
import torch

torch.set_num_threads(1)  # tiny CPU models: one thread is faster than oversubscribed BLAS threads
from torch.nn import functional as F

from mlbook.transformer.bert import BERT, BERTConfig, MLMHead, NSPHead, mask_tokens_for_mlm
from mlbook.transformer.t5 import T5, T5Config, shift_right, span_corruption


def test_mask_tokens_for_mlm_ratios():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(5, 100, (64, 128))
    inputs, labels = mask_tokens_for_mlm(ids, vocab_size=100, mask_id=4, special_ids={0, 1, 2, 3}, generator=g)
    selected = labels != -100
    frac = selected.float().mean().item()
    assert 0.13 < frac < 0.17
    masked = (inputs == 4) & selected
    changed = (inputs != ids) & selected & ~masked
    same = (inputs == ids) & selected
    n = selected.sum().item()
    assert 0.75 < masked.sum().item() / n < 0.85
    assert 0.06 < changed.sum().item() / n < 0.14
    assert 0.06 < same.sum().item() / n < 0.15
    assert torch.equal(inputs[~selected], ids[~selected])


def test_bert_is_bidirectional_and_masks_padding():
    cfg = BERTConfig(vocab_size=30, max_len=16, n_layers=2, n_heads=2, d_model=16)
    model = BERT(cfg).eval()
    ids = torch.randint(3, 30, (1, 8))
    h, pooled = model(ids)
    assert h.shape == (1, 8, 16) and pooled.shape == (1, 16)
    ids2 = ids.clone()
    ids2[0, 6] = 5
    h2, _ = model(ids2)
    assert not torch.allclose(h[:, 0], h2[:, 0])  # position 0 sees position 6 (no causal mask)
    padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    h3, _ = model(padded)
    torch.testing.assert_close(h[:, :8], h3[:, :8], atol=1e-5, rtol=1e-4)  # pads invisible


def test_mlm_head_tied_and_overfits():
    cfg = BERTConfig(vocab_size=30, max_len=16, n_layers=1, n_heads=2, d_model=16)
    model = BERT(cfg)
    head = MLMHead(cfg.d_model, model.tok_emb)
    nsp = NSPHead(cfg.d_model)
    assert head.decoder.weight is model.tok_emb.weight
    ids = torch.tensor([[7, 8, 9, 10, 11, 12]])
    inputs = ids.clone()
    inputs[0, 2] = 4  # [MASK]
    labels = torch.full_like(ids, -100)
    labels[0, 2] = 9
    params = {id(p): p for m in (model, head, nsp) for p in m.parameters()}  # dedupe the tied embedding
    opt = torch.optim.Adam(list(params.values()), lr=1e-2)
    for _ in range(80):
        h, pooled = model(inputs)
        loss = F.cross_entropy(head(h).view(-1, 30), labels.view(-1), ignore_index=-100) + F.cross_entropy(nsp(pooled), torch.tensor([1]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    h, _ = model(inputs)
    assert head(h)[0, 2].argmax().item() == 9


def test_span_corruption_structure():
    toks = list(range(10, 30))
    inp, tgt = span_corruption(toks, sentinel_start=100, noise_density=0.15, mean_span=3.0, rng=np.random.default_rng(1))
    sentinels_in = [t for t in inp if t >= 100]
    sentinels_tgt = [t for t in tgt if t >= 100]
    assert sentinels_in == sorted(sentinels_in) and len(sentinels_in) >= 1
    assert sentinels_tgt == sentinels_in + [sentinels_in[-1] + 1]  # target closes with one more sentinel
    kept = [t for t in inp if t < 100]
    dropped = [t for t in tgt if t < 100]
    assert sorted(kept + dropped) == toks


def test_shift_right():
    labels = torch.tensor([[5, 6, 7]])
    assert torch.equal(shift_right(labels, 0), torch.tensor([[0, 5, 6]]))


def test_t5_shapes_and_learns_copy():
    cfg = T5Config(vocab_size=20, n_layers=1, n_heads=2, d_model=32)
    model = T5(cfg)
    g = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(300):
        src = torch.randint(3, 20, (32, 5), generator=g)
        logits = model(src, shift_right(src, cfg.decoder_start_id))
        loss = F.cross_entropy(logits.reshape(-1, 20), src.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert logits.shape == (32, 5, 20)
    model.eval()
    src = torch.randint(3, 20, (16, 5), generator=g)
    pred = model(src, shift_right(src, 0)).argmax(-1)
    assert (pred == src).float().mean() > 0.9
