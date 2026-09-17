import torch

from mlbook.transformer.masks import apply_mask, causal_mask, causal_mask_with_cache, combine_masks, padding_mask


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    assert m.shape == (1, 1, 4, 4)
    assert torch.equal(m[0, 0], torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_causal_mask_with_cache_matches_slice_of_full_mask():
    full = causal_mask(7)[0, 0]  # (7, 7)
    part = causal_mask_with_cache(T_new=3, T_total=7)[0, 0]  # queries 4,5,6 vs keys 0..6
    assert torch.equal(part, full[4:, :])
    single = causal_mask_with_cache(1, 7)[0, 0]
    assert single.all()  # the newest token sees everything


def test_padding_mask_and_combine():
    is_pad = torch.tensor([[False, False, True], [False, True, True]])
    pm = padding_mask(is_pad)
    assert pm.shape == (2, 1, 1, 3)
    cm = combine_masks(causal_mask(3), pm, None)
    assert cm.shape == (2, 1, 3, 3)
    assert not cm[0, 0, 2, 2] and cm[0, 0, 1, 1] and not cm[1, 0, 2, 1]


def test_apply_mask_uses_finite_min_and_softmax_has_no_nan():
    scores = torch.zeros(1, 1, 2, 3, dtype=torch.float16)
    mask = torch.tensor([[[[True, False, False], [False, False, False]]]])
    out = apply_mask(scores, mask)
    assert out[0, 0, 0, 1] == torch.finfo(torch.float16).min
    probs = torch.softmax(out.float(), dim=-1)
    assert not torch.isnan(probs).any()
    torch.testing.assert_close(probs[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
