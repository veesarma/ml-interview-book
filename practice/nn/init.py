# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/nn/init.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k init -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py nn/init --force

"""Weight initialisers for the row-major convention ``Z = X W`` (Part III, chapter 4).

``W`` has shape ``(fan_in, fan_out)``: ``fan_in = W.shape[0]`` is the number of
inputs each output unit sums over, which is the quantity that controls the
forward variance (``Var(z) = fan_in * Var(w) * Var(x)``).
"""
from __future__ import annotations
from typing import Callable
import numpy as np
InitFn = Callable[[int, int, np.random.Generator], np.ndarray]

def xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Glorot & Bengio (2010): ``U(-a, a)``, ``a = sqrt(6 / (fan_in + fan_out))``.

    ``Var(w) = a^2 / 3 = 2 / (fan_in + fan_out)`` - the harmonic compromise between
    preserving forward variance (needs ``1/fan_in``) and backward variance (needs ``1/fan_out``).
    Returns ``(fan_in, fan_out)``.
    """
    raise NotImplementedError('TODO: implement xavier_uniform (see the reference in src/mlbook)')

def xavier_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """``N(0, 2 / (fan_in + fan_out))``. Returns ``(fan_in, fan_out)``."""
    raise NotImplementedError('TODO: implement xavier_normal (see the reference in src/mlbook)')

def kaiming_normal(fan_in: int, fan_out: int, rng: np.random.Generator, gain: float=np.sqrt(2.0)) -> np.ndarray:
    """He et al. (2015): ``N(0, gain^2 / fan_in)``; ``gain = sqrt 2`` for ReLU.

    ReLU zeroes half of a symmetric distribution, so ``E[relu(z)^2] = Var(z)/2``;
    the ``2`` in the numerator cancels that halving. Returns ``(fan_in, fan_out)``.
    """
    raise NotImplementedError('TODO: implement kaiming_normal (see the reference in src/mlbook)')

def kaiming_uniform(fan_in: int, fan_out: int, rng: np.random.Generator, gain: float=np.sqrt(2.0)) -> np.ndarray:
    """``U(-a, a)`` with ``a = gain * sqrt(3 / fan_in)`` so ``Var(w) = gain^2 / fan_in``."""
    raise NotImplementedError('TODO: implement kaiming_uniform (see the reference in src/mlbook)')

def lecun_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """``N(0, 1 / fan_in)``: preserves forward variance for linear / tanh / SELU nets."""
    raise NotImplementedError('TODO: implement lecun_normal (see the reference in src/mlbook)')

def orthogonal(fan_in: int, fan_out: int, rng: np.random.Generator, gain: float=1.0) -> np.ndarray:
    """Saxe et al. (2014): a (semi-)orthogonal matrix scaled by ``gain``.

    QR-decompose a Gaussian matrix and fix the signs so the distribution is
    Haar-uniform. A square orthogonal ``W`` maps every input norm exactly
    (``||xW|| = ||x||``), so depth cannot shrink or blow up the signal in a
    linear network. Returns ``(fan_in, fan_out)``.
    """
    raise NotImplementedError('TODO: implement orthogonal (see the reference in src/mlbook)')

def gpt2_residual_normal(fan_in: int, fan_out: int, rng: np.random.Generator, n_layers: int, std: float=0.02) -> np.ndarray:
    """GPT-2 scaled init for the projection that writes into the residual stream.

    ``N(0, (std / sqrt(2 n_layers))^2)``: with ``2 n_layers`` residual branches
    (attention + MLP per block) each adding variance ``sigma^2`` to the stream,
    the sum has variance ``2 L sigma^2``; the ``1 / sqrt(2L)`` keeps it at
    ``sigma^2`` regardless of depth. Returns ``(fan_in, fan_out)``.
    """
    raise NotImplementedError('TODO: implement gpt2_residual_normal (see the reference in src/mlbook)')

def zeros(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """All-zero weights (Fixup / zero-gamma style: the residual branch starts as identity)."""
    raise NotImplementedError('TODO: implement zeros (see the reference in src/mlbook)')

def activation_std_by_depth(init: InitFn, activation: Callable[[np.ndarray], np.ndarray], width: int=256, depth: int=20, n_samples: int=512, seed: int=0) -> np.ndarray:
    """Push ``N(0,1)`` inputs through ``depth`` layers; return the std after each layer.

    Returns ``(depth,)``. Used by the chapter-4 figure and the variance tests.
    """
    raise NotImplementedError('TODO: implement activation_std_by_depth (see the reference in src/mlbook)')
