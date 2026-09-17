import math

import numpy as np

from mlbook.retrieval.bm25 import BM25, tokenize

DOCS = ["the cat sat on the mat", "the dog chased the cat", "a bird flew over the house", "cat cat cat"]


def test_tokenize():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_bm25_idf_matches_formula():
    bm = BM25().fit(DOCS)
    n_cat = 3
    assert math.isclose(bm.idf["cat"], math.log((4 - n_cat + 0.5) / (n_cat + 0.5) + 1))
    assert bm.idf["bird"] > bm.idf["cat"]  # n=1 vs n=3
    assert bm.idf["cat"] == bm.idf["the"]  # both in 3 documents


def test_bm25_term_score_saturates_in_tf():
    bm = BM25(k1=1.2, b=0.0).fit(DOCS)
    s = [bm.term_score(tf, 5.0) for tf in (1, 2, 4, 8, 100)]
    assert all(s[i] < s[i + 1] for i in range(len(s) - 1))
    assert s[-1] < bm.k1 + 1.0 + 1e-9  # upper bound (k1 + 1)


def test_bm25_hand_computed_score():
    bm = BM25(k1=1.5, b=0.75).fit(DOCS)
    # document 3 "cat cat cat": tf=3, len=3, avgdl=(6+5+6+3)/4=5
    avgdl = 5.0
    denom = 3 + 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl)
    expected = bm.idf["cat"] * 3 * 2.5 / denom
    assert np.isclose(bm.score("cat")[3], expected)


def test_bm25_topk_prefers_rare_terms():
    bm = BM25().fit(DOCS)
    ids, _ = bm.topk("bird house", 2)
    assert ids[0] == 2
    assert bm.score("zzz unknown").sum() == 0.0
