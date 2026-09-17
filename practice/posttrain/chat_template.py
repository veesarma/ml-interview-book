# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/chat_template.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k chat_template -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/chat_template --force

"""Chat templates and a toy tokenizer for the post-training chapters.

A chat template turns a list of ``{"role", "content"}`` messages into one token
sequence with *role markers* (special tokens) so the model can tell who said
what. The template below is a ChatML-style layout::

    <|system|> You are a helpful assistant. <|end|>
    <|user|> 2 + 3 = <|end|>
    <|assistant|> 5 <|end|>

``tokenize_chat`` additionally returns an ``assistant_mask`` marking exactly the
tokens the SFT loss should be computed on: the assistant's content **and** its
closing ``<|end|>`` (the model must learn when to stop), nothing else.
"""
from __future__ import annotations
from dataclasses import dataclass, field
SPECIAL_TOKENS: tuple[str, ...] = ('<|pad|>', '<|end|>', '<|system|>', '<|user|>', '<|assistant|>', '<|unk|>')
DIGITS: tuple[str, ...] = tuple((str(i) for i in range(10)))
WORDS: tuple[str, ...] = ('+', '-', '=', '?', '.', ',', 'you', 'are', 'a', 'an', 'the', 'helpful', 'assistant', 'answer', 'briefly', 'what', 'is', 'sum', 'of', 'and', 'yes', 'no', 'hello', 'thanks', 'sorry', 'please', 'I', 'think', 'so', 'not', 'sure', 'good', 'bad', 'great', 'cat', 'dog', 'blue', 'red')

@dataclass
class ToyTokenizer:
    """Whitespace tokenizer over a fixed vocabulary of specials + digits + a few words.

    ``vocab_size`` is 6 specials + 10 digits + len(WORDS) = 54.
    """
    tokens: tuple[str, ...] = field(default_factory=lambda: SPECIAL_TOKENS + DIGITS + WORDS)

    def __post_init__(self) -> None:
        raise NotImplementedError('TODO: implement __post_init__ (see the reference in src/mlbook)')

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError('TODO: implement vocab_size (see the reference in src/mlbook)')

    def encode(self, text: str) -> list[int]:
        """Split on whitespace and map each piece to an id (unknown pieces -> ``<|unk|>``)."""
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool=False) -> str:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

def render_chat(messages: list[dict[str, str]], add_generation_prompt: bool=False) -> str:
    """Render messages as text using the ChatML-style template.

    Args:
        messages: list of ``{"role": "system"|"user"|"assistant", "content": str}``.
        add_generation_prompt: append an opening ``<|assistant|>`` so the model
            continues as the assistant (used at inference time).
    Returns:
        the templated string, one message per line.
    """
    raise NotImplementedError('TODO: implement render_chat (see the reference in src/mlbook)')

def tokenize_chat(messages: list[dict[str, str]], tok: ToyTokenizer) -> tuple[list[int], list[int]]:
    """Tokenize a conversation and mark the tokens the SFT loss should train on.

    Returns:
        input_ids: list of length ``T``.
        assistant_mask: list of length ``T`` with 1 on assistant content tokens and
            on the ``<|end|>`` that closes an assistant turn, 0 on everything else
            (system/user text, role markers, user ``<|end|>``).
    """
    raise NotImplementedError('TODO: implement tokenize_chat (see the reference in src/mlbook)')

def prompt_ids(messages: list[dict[str, str]], tok: ToyTokenizer) -> list[int]:
    """Token ids of a conversation ending in an open assistant turn (what the policy samples from)."""
    raise NotImplementedError('TODO: implement prompt_ids (see the reference in src/mlbook)')
