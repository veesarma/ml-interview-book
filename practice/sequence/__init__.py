"""Sequence models before the Transformer: RNN, LSTM, GRU, seq2seq with attention.

NumPy for the recurrences whose backward pass you should be able to derive by hand
(``rnn``, ``lstm``); PyTorch where autograd and batching matter (``gru``,
``seq2seq_attention``).
"""

from . import gru, lstm, rnn, seq2seq_attention

__all__ = ["rnn", "lstm", "gru", "seq2seq_attention"]
