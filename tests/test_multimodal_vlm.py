import torch

from mlbook.multimodal.vlm import MiniVLM, TinyCausalLM, ToyVisionEncoder, merge_visual_tokens, vlm_lm_loss

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers


def _vlm(vocab=16, d_v=16, d_llm=24, image_token_id=15):
    vision = ToyVisionEncoder(image_size=8, patch=4, in_channels=1, d_v=d_v, depth=1, n_heads=2)
    lm = TinyCausalLM(vocab=vocab, max_len=32, d_llm=d_llm, depth=2, n_heads=4)
    return MiniVLM(vision, d_v, lm, d_llm, image_token_id)


def test_merge_and_forward_shapes():
    model = _vlm()
    imgs = torch.randn(2, 1, 8, 8)
    ids = torch.tensor([[1, 15, 3, 4, 5], [2, 15, 6, 7, 8]])  # (B, T=5), image token at col 1
    logits, is_visual = model(imgs, ids)
    assert logits.shape == (2, 5 - 1 + 4, 16)
    assert is_visual.sum(1).tolist() == [4, 4]
    assert bool(is_visual[0, 1]) and bool(is_visual[0, 4]) and not bool(is_visual[0, 5])
    text = model.lm.tok(ids)
    merged, pos, _ = merge_visual_tokens(text, ids, torch.zeros(2, 4, 24), 15)
    assert torch.equal(merged[:, 0], text[:, 0]) and torch.equal(merged[:, 5], text[:, 2])
    assert pos.shape == (2, 8) and pos[0, -1] == 7


def test_vlm_overfits_captions():
    torch.manual_seed(0)
    model = _vlm()
    n = 8
    imgs = torch.randn(n, 1, 8, 8) * 0.2
    ids = torch.zeros(n, 5, dtype=torch.long)
    for i in range(n):
        k = i % 4
        r, c = (k // 2) * 4, (k % 2) * 4
        imgs[i, 0, r : r + 4, c : c + 4] += 3.0
        ids[i] = torch.tensor([1, 15, 3, 4 + k, 4 + k])  # caption depends on the image
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(120):
        opt.zero_grad()
        logits, is_visual = model(imgs, ids)
        loss = vlm_lm_loss(logits, ids, is_visual, 15)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()
    logits, is_visual = model(imgs, ids)
    # position that predicts the class-dependent token: merged index of ids[:, 3] is 1 + 4 + 1 = 6 -> pred at 5
    pred = logits[:, 5].argmax(-1)
    assert torch.equal(pred, ids[:, 3])


def test_causal_lm_is_causal():
    lm = TinyCausalLM(vocab=16, max_len=32, d_llm=24, depth=2, n_heads=4)
    ids = torch.randint(0, 16, (1, 6))
    pos = torch.arange(6)[None]
    a = lm(lm.tok(ids), pos)
    ids2 = ids.clone(); ids2[0, 4] = (ids2[0, 4] + 1) % 16  # change token 4
    b = lm(lm.tok(ids2), pos)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5)  # positions < 4 unaffected
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_vlm_lm_loss_ignores_visual_positions():
    B, L, V = 1, 8, 16
    logits = torch.randn(B, L, V)
    ids = torch.tensor([[1, 15, 3, 4, 5]])  # image token at column 1 expands to N_v = 4
    is_visual = torch.tensor([[False, True, True, True, True, False, False, False]])
    loss = vlm_lm_loss(logits, ids, is_visual, 15)
    # manual: targets = [1, -100 x4, 3, 4, 5]; predictions at positions 4, 5, 6 predict 3, 4, 5
    lp = logits[0].log_softmax(-1)
    manual = -(lp[4, 3] + lp[5, 4] + lp[6, 5]) / 3
    assert torch.isclose(loss, manual)
