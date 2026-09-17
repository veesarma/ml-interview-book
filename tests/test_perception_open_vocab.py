import torch

from mlbook.perception import open_vocab as ov


def test_tiny_text_encoder_is_normalised_and_masks_padding():
    torch.manual_seed(0)
    enc = ov.TinyTextEncoder(vocab_size=50, d_embed=16)
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    e = enc(ids, mask)
    assert e.shape == (2, 16)
    assert torch.allclose(e.norm(dim=-1), torch.ones(2), atol=1e-5)
    ids2 = ids.clone()
    ids2[1, 3] = 9  # changing a padded token must not change the embedding
    assert torch.allclose(enc(ids2, mask), e)


def test_open_vocab_head_recovers_classes_and_adds_new_ones_without_training():
    torch.manual_seed(0)
    d = 8
    head = ov.OpenVocabHead(d_region=d, d_embed=d, bias_init=0.0)
    with torch.no_grad():  # identity projection so region features live in the text space
        head.proj.weight.copy_(torch.eye(d))
        head.proj.bias.zero_()
    text = torch.eye(d)[:3]  # (K=3, d) three orthogonal "phrases"
    regions = text[[2, 0, 1]].unsqueeze(0) + 0.05 * torch.randn(1, 3, d)  # (B=1, N=3, d)
    logits = head(regions, text)
    assert logits.shape == (1, 3, 3)
    labels, is_obj = ov.zero_shot_classify(logits)
    assert labels[0].tolist() == [2, 0, 1] and is_obj.all()
    # Open vocabulary: add a fourth phrase — a new row, no gradient step — and a region that matches it.
    text4 = torch.eye(d)[:4]
    regions4 = torch.cat([regions, torch.eye(d)[3].view(1, 1, d)], dim=1)
    labels4, _ = ov.zero_shot_classify(head(regions4, text4))
    assert labels4[0].tolist() == [2, 0, 1, 3]


def test_region_word_alignment_loss_decreases_with_training():
    torch.manual_seed(0)
    head = ov.OpenVocabHead(d_region=12, d_embed=8)
    text = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    regions = torch.randn(4, 6, 12)
    targets = torch.zeros(4, 6, 5)
    targets[:, torch.arange(6), torch.arange(6) % 5] = 1.0  # one phrase per region
    targets[:, 5] = 0.0  # last region is background
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    first = ov.region_word_alignment_loss(head(regions, text), targets).item()
    for _ in range(60):
        opt.zero_grad()
        loss = ov.region_word_alignment_loss(head(regions, text), targets)
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first
    labels, is_obj = ov.zero_shot_classify(head(regions, text))
    assert labels[:, :5].tolist() == [[0, 1, 2, 3, 4]] * 4
    assert not is_obj[:, 5].any()  # background regions stay below threshold
