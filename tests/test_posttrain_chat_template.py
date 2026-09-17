"""Chat template + toy tokenizer: role markers, assistant mask, round trips."""
import torch

from mlbook.posttrain.chat_template import ToyTokenizer, prompt_ids, render_chat, tokenize_chat

MSGS = [
    {"role": "system", "content": "you are a helpful assistant ."},
    {"role": "user", "content": "2 + 3 ="},
    {"role": "assistant", "content": "5"},
]


def test_render_chat_matches_template():
    text = render_chat(MSGS)
    assert text == "<|system|> you are a helpful assistant . <|end|>\n<|user|> 2 + 3 = <|end|>\n<|assistant|> 5 <|end|>"
    assert render_chat(MSGS[:2], add_generation_prompt=True).endswith("\n<|assistant|>")


def test_tokenize_chat_marks_only_assistant_tokens():
    tok = ToyTokenizer()
    ids, mask = tokenize_chat(MSGS, tok)
    assert len(ids) == len(mask)
    # assistant content "5" and its closing <|end|> are the only mask==1 tokens
    marked = [tok.tokens[i] for i, m in zip(ids, mask) if m == 1]
    assert marked == ["5", "<|end|>"]
    # the assistant role marker itself is not a target
    assert mask[ids.index(tok.role_ids["assistant"])] == 0
    # user/system tokens are all unmasked
    assert sum(mask) == 2


def test_encode_decode_round_trip_and_unknown():
    tok = ToyTokenizer()
    assert tok.decode(tok.encode("hello 7 + 2 =")) == "hello 7 + 2 ="
    assert tok.encode("zebra") == [tok.unk_id]
    assert tok.vocab_size == len(set(tok.tokens))


def test_prompt_ids_ends_with_open_assistant_turn():
    tok = ToyTokenizer()
    ids = prompt_ids(MSGS[:2], tok)
    assert ids[-1] == tok.role_ids["assistant"]
    assert ids[-2] == tok.eos_id  # user turn closed before assistant opens
    assert torch.tensor(ids).dtype == torch.long
