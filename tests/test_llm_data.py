"""Tests for data_dedup, packing and lm_loss (Part VI, chapter 1)."""

import numpy as np
import torch
import torch.nn.functional as F

from mlbook.llm.data_dedup import (
    LSHIndex,
    MinHash,
    estimate_jaccard,
    exact_dedup,
    jaccard,
    near_dedup,
    ngram_contamination,
    shingles,
)
from mlbook.llm.lm_loss import bits_per_byte, causal_mask, next_token_loss, per_token_nll
from mlbook.llm.packing import document_causal_mask, pack_sequences, pad_rows, position_ids_within_document

DOC_A = "the quick brown fox jumps over the lazy dog and then runs into the forest to hide from the hunter"
DOC_A_EDIT = "the quick brown fox jumps over the lazy dog and then runs into the woods to hide from the hunter"
DOC_B = "large language models are trained on trillions of tokens scraped from the public web and books"


def test_shingles_count():
    assert len(shingles("a b c d e f", n=5)) == 2
    assert shingles("a b", n=5) == {"a b"}


def test_exact_dedup_keeps_first():
    assert exact_dedup([DOC_A, DOC_B, DOC_A]) == [0, 1]


def test_minhash_estimates_jaccard():
    mh = MinHash(num_perm=256)
    s1, s2 = shingles(DOC_A, 3), shingles(DOC_A_EDIT, 3)
    exact = jaccard(s1, s2)
    est = estimate_jaccard(mh.signature(s1), mh.signature(s2))
    assert abs(est - exact) < 0.12  # std of the estimate is ~sqrt(J(1-J)/256) ≈ 0.03
    assert estimate_jaccard(mh.signature(s1), mh.signature(shingles(DOC_B, 3))) < 0.1


def test_lsh_index_finds_near_duplicates_only():
    mh = MinHash(num_perm=128)
    idx = LSHIndex(num_perm=128, bands=16)  # rows = 8
    sig_a, sig_b = mh.signature(shingles(DOC_A, 3)), mh.signature(shingles(DOC_B, 3))
    idx.add(0, sig_a)
    idx.add(1, sig_b)
    assert 0 in idx.query(mh.signature(shingles(DOC_A_EDIT, 3)))
    assert idx.query(mh.signature(shingles("completely unrelated text about cooking pasta al dente", 3))) == set()
    assert abs(idx.threshold() - (1 / 16) ** (1 / 8)) < 1e-9  # ≈ 0.71


def test_near_dedup_pipeline():
    docs = [DOC_A, DOC_B, DOC_A_EDIT, DOC_A]
    assert near_dedup(docs, n=3, num_perm=128, bands=32, threshold=0.5) == [0, 1]


def test_ngram_contamination():
    train = [DOC_A]
    evals = [DOC_A_EDIT, DOC_B]
    res = ngram_contamination(train, evals, n=8)
    assert res["contaminated_indices"] == [0]
    assert res["contaminated_fraction"] == 0.5


def test_pack_sequences_first_fit_and_doc_ids():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11, 12]]
    rows, doc_ids = pack_sequences(docs, max_len=8, eos_id=0)
    assert rows[0] == [1, 2, 3, 0, 4, 5, 0, 6]  # doc 2 starts in the leftover slot
    assert doc_ids[0] == [0, 0, 0, 0, 1, 1, 1, 2]
    assert rows[1] == [7, 8, 9, 10, 11, 12, 0]
    assert sum(len(r) for r in rows) == sum(len(d) + 1 for d in docs)


def test_document_causal_mask_blocks_cross_document():
    doc_ids = torch.tensor([[0, 0, 1, 1]])
    m = document_causal_mask(doc_ids)[0]
    expected = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(m, expected)
    assert torch.equal(position_ids_within_document(doc_ids), torch.tensor([[0, 1, 0, 1]]))
    padded = pad_rows([[1, 2], [3]], max_len=3, pad_id=-1)
    assert padded.tolist() == [[1, 2, -1], [3, -1, -1]]


def test_next_token_loss_matches_cross_entropy_with_shift():
    B, T, V = 2, 6, 11
    logits = torch.randn(B, T, V)
    tokens = torch.randint(0, V, (B, T))
    tokens[1, -1] = -100  # padded target
    ours = next_token_loss(logits, tokens)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(ours, ref, atol=1e-6)
    assert per_token_nll(logits, tokens.clamp(min=0)).shape == (B, T - 1)


def test_causal_mask_and_bits_per_byte():
    assert torch.equal(causal_mask(3), torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    # 100 tokens at ln2 nats each = 100 bits; over 200 bytes = 0.5 bpb
    assert abs(bits_per_byte(100 * np.log(2), 200) - 0.5) < 1e-9
