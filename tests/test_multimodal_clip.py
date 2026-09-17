import math

import torch

from mlbook.multimodal.clip import MiniCLIP, TinyImageEncoder, TinyTextEncoder, clip_loss, siglip_loss, zero_shot_classify

torch.set_num_threads(1)  # multi-threaded CPU kernels are pathologically slow on tiny tensors in CI containers


def _toy_data(n_classes: int = 4, per_class: int = 4, seed: int = 0):
    """Class k image: a bright k-th 4x4 quadrant; class k caption: token ids [1, k+2, 0(pad)]."""
    g = torch.Generator().manual_seed(seed)
    imgs, ids, ys = [], [], []
    for k in range(n_classes):
        for _ in range(per_class):
            x = torch.randn(1, 8, 8, generator=g) * 0.3
            r, c = (k // 2) * 4, (k % 2) * 4
            x[0, r : r + 4, c : c + 4] += 3.0
            imgs.append(x)
            ids.append(torch.tensor([1, k + 2, 0]))
            ys.append(k)
    return torch.stack(imgs), torch.stack(ids), torch.tensor(ys)


def _model():
    img = TinyImageEncoder(image_size=8, patch=4, in_channels=1, d=32, depth=1, n_heads=4)
    txt = TinyTextEncoder(vocab=8, max_len=4, d=32, depth=1, n_heads=4)
    return MiniCLIP(img, txt, d_img=32, d_txt=32, d_embed=16)


def test_loss_at_init_is_log_batch():
    B = 6
    logits = torch.zeros(B, B)
    assert math.isclose(clip_loss(logits).item(), math.log(B), rel_tol=1e-6)


def test_siglip_loss_matches_manual():
    v = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    t = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    log_t, b = torch.tensor(math.log(10.0)), torch.tensor(-10.0)
    loss = siglip_loss(v, t, log_t, b)
    manual = 0.0
    for i in range(3):
        for j in range(3):
            z = 1.0 if i == j else -1.0
            logit = 10.0 * float(v[i] @ t[j]) - 10.0
            manual += -math.log(1 / (1 + math.exp(-z * logit)))
    assert math.isclose(loss.item(), manual / 3, rel_tol=1e-5)


def test_mini_clip_learns_alignment_and_zero_shot():
    imgs, ids, ys = _toy_data()
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    # batch = one image per class (unique captions) so the diagonal is the only positive
    for step in range(80):
        sel = torch.tensor([k * 4 + (step % 4) for k in range(4)])
        opt.zero_grad()
        loss = clip_loss(model(imgs[sel], ids[sel]))
        loss.backward()
        opt.step()
    prompts = torch.stack([torch.tensor([1, k + 2, 0]) for k in range(4)])  # (K, T)
    acc = (zero_shot_classify(model, imgs, prompts).argmax(-1) == ys).float().mean().item()
    assert acc > 0.9, acc
    assert model.logit_scale.item() <= math.log(100.0) + 1e-6


def test_siglip_training_also_aligns():
    imgs, ids, ys = _toy_data(seed=1)
    model = _model()
    log_t = torch.nn.Parameter(torch.tensor(math.log(10.0)))
    bias = torch.nn.Parameter(torch.tensor(-10.0))
    opt = torch.optim.Adam(list(model.parameters()) + [log_t, bias], lr=3e-3)
    for step in range(80):
        sel = torch.tensor([k * 4 + (step % 4) for k in range(4)])
        opt.zero_grad()
        loss = siglip_loss(model.encode_image(imgs[sel]), model.encode_text(ids[sel]), log_t, bias)
        loss.backward()
        opt.step()
    prompts = torch.stack([torch.tensor([1, k + 2, 0]) for k in range(4)])
    acc = (zero_shot_classify(model, imgs, prompts).argmax(-1) == ys).float().mean().item()
    assert acc > 0.9, acc


def test_text_encoder_ignores_padding():
    enc = TinyTextEncoder(vocab=8, max_len=4, d=16, depth=1, n_heads=2)
    a = enc(torch.tensor([[1, 2, 0, 0]]))
    b = enc(torch.tensor([[1, 2, 0, 5]]))  # token after a pad is itself masked only if it is pad
    c = enc(torch.tensor([[1, 2, 3, 0]]))
    assert a.shape == (1, 16)
    assert not torch.allclose(a, b)  # token 5 is real and changes the output
    assert not torch.allclose(a, c)


def test_image_encoder_shape():
    enc = TinyImageEncoder(image_size=8, patch=4, in_channels=1, d=16, depth=1, n_heads=2)
    assert enc(torch.randn(3, 1, 8, 8)).shape == (3, 16)
