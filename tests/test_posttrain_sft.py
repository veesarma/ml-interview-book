"""SFT: assistant-only labels, masked loss, packing with segment ids, and the training loop."""
import torch
import torch.nn.functional as F

from mlbook.posttrain.chat_template import ToyTokenizer, tokenize_chat
from mlbook.posttrain.sft import (
    IGNORE_INDEX,
    PackedBatch,
    assistant_only_labels,
    pack_examples,
    packed_attention_mask,
    sft_loss,
    train_sft,
)
from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig


def _example(tok, user, answer):
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    return tokenize_chat(msgs, tok)


def test_assistant_only_labels_shift_and_mask():
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    amask = torch.tensor([[0, 0, 1, 1, 0]])  # tokens 12, 13 are assistant tokens
    labels = assistant_only_labels(ids, amask)
    # position t predicts token t+1: only positions 1 (->12) and 2 (->13) are targets
    assert labels.tolist() == [[IGNORE_INDEX, 12, 13, IGNORE_INDEX, IGNORE_INDEX]]


def test_sft_loss_excludes_masked_tokens():
    torch.manual_seed(0)
    B, T, V = 2, 6, 11
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = IGNORE_INDEX  # first three positions are "user" tokens
    loss = sft_loss(logits, labels)
    # manual: mean CE over the unmasked positions only
    manual = F.cross_entropy(logits[:, 3:, :].reshape(-1, V), labels[:, 3:].reshape(-1))
    assert torch.allclose(loss, manual)
    # changing logits at masked positions must not change the loss
    logits2 = logits.clone()
    logits2[:, :3, :] += 100.0
    assert torch.allclose(sft_loss(logits2, labels), loss)


def test_pack_examples_segments_and_labels():
    tok = ToyTokenizer()
    ex = [_example(tok, "hello", "hello"), _example(tok, "2 + 2 =", "4"), _example(tok, "thanks", "good")]
    batch = pack_examples(ex, max_len=16, pad_id=tok.pad_id)
    assert isinstance(batch, PackedBatch)
    assert batch.input_ids.shape == batch.labels.shape == batch.segment_ids.shape
    assert batch.input_ids.shape[1] == 16
    # segments are numbered 1.. within a row, padding is 0
    seg = batch.segment_ids[0]
    assert seg[0] == 1 and int(seg.max()) >= 2 and int(seg[-1]) == 0
    # no label crosses a segment boundary: the last token of a segment has IGNORE label
    for b in range(seg.shape[0] if seg.dim() > 1 else 1):
        row_seg = batch.segment_ids[b]
        for t in range(15):
            if row_seg[t] != row_seg[t + 1]:
                assert batch.labels[b, t] == IGNORE_INDEX
    # the number of labelled targets equals the number of assistant tokens across examples
    n_targets = int((batch.labels != IGNORE_INDEX).sum())
    assert n_targets == sum(sum(m) for _, m in ex)


def test_packed_attention_mask_blocks_cross_example_attention():
    seg = torch.tensor([[1, 1, 1, 2, 2, 0]])
    allowed = packed_attention_mask(seg)
    assert allowed.shape == (1, 6, 6)
    assert allowed[0, 4, 3] and allowed[0, 2, 0]  # within-segment, causal
    assert not allowed[0, 3, 2]  # segment 2 may not see segment 1
    assert not allowed[0, 0, 1]  # causal
    assert allowed[0, 5, 5]  # padding attends to itself (no all-masked softmax row)


def test_packed_mask_makes_examples_independent():
    """Logits for example 2 must equal the logits it would get alone (no leakage from example 1)."""
    torch.manual_seed(0)
    tok = ToyTokenizer()
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, max_len=32)
    model = TinyCausalLM(cfg).eval()
    ex1, ex2 = _example(tok, "hello", "hello"), _example(tok, "2 + 2 =", "4")
    packed = pack_examples([ex1, ex2], max_len=24, pad_id=tok.pad_id)
    allowed = packed_attention_mask(packed.segment_ids)
    with torch.no_grad():
        packed_logits = model(packed.input_ids, allowed)[0]  # (T, V)
        alone_logits = model(torch.tensor([ex2[0]]))[0]  # (T2, V)
    start = len(ex1[0])
    # Positions differ (learned positional embeddings), so compare after removing that effect:
    # with pos-embedding zeroed the two must match exactly.
    model.pos_emb.weight.data.zero_()
    with torch.no_grad():
        packed_logits = model(packed.input_ids, allowed)[0]
        alone_logits = model(torch.tensor([ex2[0]]))[0]
    assert torch.allclose(packed_logits[start:start + len(ex2[0])], alone_logits, atol=1e-5)


def test_train_sft_reduces_loss_and_learns_answer():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=2, max_len=32)
    model = TinyCausalLM(cfg)
    ex = [_example(tok, "what is the sum of 1 and 1 ?", "2"), _example(tok, "hello", "hello")]
    batch = pack_examples(ex, max_len=24, pad_id=tok.pad_id)
    losses = train_sft(model, [batch], epochs=60, lr=1e-2)
    assert losses[-1] < 0.3 * losses[0]
    # the model now predicts "2" after "<|assistant|>" for the first prompt
    ids, _ = ex[0]
    pos = ids.index(tok.role_ids["assistant"])
    with torch.no_grad():
        pred = model(torch.tensor([ids]))[0, pos].argmax()
    assert tok.tokens[int(pred)] == "2"
