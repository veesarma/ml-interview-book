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

SPECIAL_TOKENS: tuple[str, ...] = ("<|pad|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>", "<|unk|>")
DIGITS: tuple[str, ...] = tuple(str(i) for i in range(10))
WORDS: tuple[str, ...] = (
    "+", "-", "=", "?", ".", ",", "you", "are", "a", "an", "the", "helpful", "assistant", "answer",
    "briefly", "what", "is", "sum", "of", "and", "yes", "no", "hello", "thanks", "sorry", "please",
    "I", "think", "so", "not", "sure", "good", "bad", "great", "cat", "dog", "blue", "red",
)


@dataclass
class ToyTokenizer:
    """Whitespace tokenizer over a fixed vocabulary of specials + digits + a few words.

    ``vocab_size`` is 6 specials + 10 digits + len(WORDS) = 54.
    """

    tokens: tuple[str, ...] = field(default_factory=lambda: SPECIAL_TOKENS + DIGITS + WORDS)

    def __post_init__(self) -> None:
        self.id_of: dict[str, int] = {tok: i for i, tok in enumerate(self.tokens)}
        self.pad_id = self.id_of["<|pad|>"]
        self.eos_id = self.id_of["<|end|>"]
        self.unk_id = self.id_of["<|unk|>"]
        self.role_ids = {role: self.id_of[f"<|{role}|>"] for role in ("system", "user", "assistant")}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Split on whitespace and map each piece to an id (unknown pieces -> ``<|unk|>``)."""
        return [self.id_of.get(piece, self.unk_id) for piece in text.split()]

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = False) -> str:
        pieces = [self.tokens[i] for i in ids]
        if skip_special:
            pieces = [p for p in pieces if p not in SPECIAL_TOKENS]
        return " ".join(pieces)


def render_chat(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render messages as text using the ChatML-style template.

    Args:
        messages: list of ``{"role": "system"|"user"|"assistant", "content": str}``.
        add_generation_prompt: append an opening ``<|assistant|>`` so the model
            continues as the assistant (used at inference time).
    Returns:
        the templated string, one message per line.
    """
    lines: list[str] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unknown role {role!r}")
        lines.append(f"<|{role}|> {content} <|end|>")
    if add_generation_prompt:
        lines.append("<|assistant|>")
    return "\n".join(lines)


def tokenize_chat(messages: list[dict[str, str]], tok: ToyTokenizer) -> tuple[list[int], list[int]]:
    """Tokenize a conversation and mark the tokens the SFT loss should train on.

    Returns:
        input_ids: list of length ``T``.
        assistant_mask: list of length ``T`` with 1 on assistant content tokens and
            on the ``<|end|>`` that closes an assistant turn, 0 on everything else
            (system/user text, role markers, user ``<|end|>``).
    """
    input_ids: list[int] = []
    assistant_mask: list[int] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        input_ids.append(tok.role_ids[role])
        assistant_mask.append(0)  # the role marker is *given*, never predicted as a target
        body = tok.encode(content)
        input_ids.extend(body)
        assistant_mask.extend([1 if role == "assistant" else 0] * len(body))
        input_ids.append(tok.eos_id)
        assistant_mask.append(1 if role == "assistant" else 0)
    return input_ids, assistant_mask


def prompt_ids(messages: list[dict[str, str]], tok: ToyTokenizer) -> list[int]:
    """Token ids of a conversation ending in an open assistant turn (what the policy samples from)."""
    ids, _ = tokenize_chat(messages, tok)
    return ids + [tok.role_ids["assistant"]]
