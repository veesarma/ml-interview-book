from mlbook.transformer.tokenizer_bpe import BPETokenizer, ByteLevelBPE, apply_merges, bytes_to_unicode, count_pairs, learn_merges, merge_pair
from mlbook.transformer.tokenizer_wordpiece import WordPieceTokenizer, unigram_em_step, unigram_prune, viterbi_segment

CORPUS = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"


def test_count_pairs_and_merge_pair():
    wf = {("l", "o", "w</w>"): 5, ("l", "o", "w", "e", "r</w>"): 2}
    pairs = count_pairs(wf)
    assert pairs[("l", "o")] == 7 and pairs[("o", "w")] == 2
    assert merge_pair(("a", "b", "a", "b", "c"), ("a", "b")) == ("ab", "ab", "c")


def test_learn_merges_reproduces_sennrich_example():
    wf = {("l", "o", "w</w>"): 5, ("l", "o", "w", "e", "r</w>"): 2, ("n", "e", "w", "e", "s", "t</w>"): 6, ("w", "i", "d", "e", "s", "t</w>"): 3}
    merges = learn_merges(wf, 3)
    # (e, s) and (s, t</w>) both occur 9 times; either order yields the symbol "est</w>"
    assert set(merges[:2]) in ({("e", "s"), ("es", "t</w>")}, {("s", "t</w>"), ("e", "st</w>")})
    assert merges[2] == ("l", "o")  # 7 occurrences, the next most frequent


def test_apply_merges_uses_merge_order():
    ranks = {("a", "b"): 0, ("ab", "c"): 1, ("b", "c"): 2}
    assert apply_merges(("a", "b", "c"), ranks) == ("abc",)
    assert apply_merges(("x", "b", "c"), ranks) == ("x", "bc")


def test_bpe_tokenizer_roundtrip_and_unseen_word():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=30)
    text = "lowest newest wide"
    assert tok.decode(tok.encode(text)) == text
    pieces = tok.tokenize("lowest")  # unseen word, split into learned subwords (merge order decides where)
    assert len(pieces) == 2 and "".join(pieces) == "lowest</w>"
    assert tok.encode("l0w")[1] == tok.vocab["<unk>"]  # '0' never seen in training


def test_bytes_to_unicode_is_bijection():
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256


def test_byte_level_bpe_roundtrips_any_unicode():
    tok = ByteLevelBPE()
    tok.train("hello world hello there héllo wörld 123 45", vocab_size=300)
    for s in ("hello world", "héllo wörld!", "日本語 テキスト", "  spaces  and\ttabs\n", "12345"):
        assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode("hello")) < 5  # merges learned


def test_wordpiece_train_encode_decode():
    wp = WordPieceTokenizer()
    wp.train(CORPUS, vocab_size=40)
    assert "[UNK]" in wp.vocab
    pieces = wp.tokenize("lowest")
    assert pieces[0] == "low" or pieces[0].startswith("l")
    assert all(p.startswith("##") for p in pieces[1:])
    assert wp.decode(wp.encode("newest widest")) == "newest widest"
    assert wp.tokenize("xyz") == ["[UNK]"]


def test_wordpiece_scores_prefer_rare_pairs():
    """Score count(ab)/(count(a)count(b)) picks a pair made of rare symbols over a frequent-but-common one."""
    wp = WordPieceTokenizer()
    wp.train("aa aa aa aa aa aa aa aa zq", vocab_size=6)  # vocab: [UNK], a, ##a, z, ##q, + 1 merge
    assert "zq" in wp.vocab  # score(z,##q) = 1/(1*1) beats score(a,##a) = 8/(8*8)


def test_unigram_viterbi_and_em():
    import math

    lp = {"h": math.log(0.1), "e": math.log(0.1), "l": math.log(0.1), "o": math.log(0.1), "he": math.log(0.2), "llo": math.log(0.3), "hello": math.log(0.1)}
    pieces, score = viterbi_segment("hello", lp)
    assert pieces == ["hello"] or pieces == ["he", "llo"]
    assert math.isclose(score, max(lp["hello"], lp["he"] + lp["llo"]))
    new = unigram_em_step({"hello": 3, "he": 2}, lp)
    assert set(new) <= set(lp) and all(v <= 0 for v in new.values())
    pruned = unigram_prune({"hello": 3, "he": 2}, lp, keep_fraction=0.5)
    assert {"h", "e", "l", "o"} <= set(pruned) and len(pruned) < len(lp)
