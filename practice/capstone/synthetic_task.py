# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/synthetic_task.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k synthetic_task -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/synthetic_task --force

"""Canon #60, part 0 -- the synthetic grounded world the capstone is trained on.

The whole capstone needs a task with three properties, and every real
post-training pipeline needs the same three:

1. **Cheap.** Images and questions are generated on the fly, so nothing is
   downloaded and every stage runs on a CPU in seconds.
2. **Deterministic.** Everything is drawn from an explicit
   :class:`numpy.random.Generator`, so a seed reproduces the dataset exactly.
3. **Verifiable.** The answer to every question is *computed from the scene
   graph that drew the image*, not labelled by a human or a judge model. That is
   what makes the reward in :mod:`mlbook.capstone.reward_stage` a **verifiable**
   reward (RLVR) rather than a learned proxy.

The world is a ``GRID x GRID`` board of ``CELL x CELL`` pixel cells. Each cell
holds at most one coloured shape, so one image patch (``CELL`` pixels wide)
contains exactly one cell. Counting a colour is therefore a sum over patches --
a task a two-block attention encoder can actually represent, which is the point.

Image tensors are ``(3, IMAGE_SIZE, IMAGE_SIZE)`` float32 in ``[0, 1]``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
GRID = 4
CELL = 4
IMAGE_SIZE = GRID * CELL
N_CELLS = GRID * GRID
COLOURS: tuple[str, ...] = ('red', 'green', 'blue')
CHANNEL: dict[str, int] = {'red': 0, 'green': 1, 'blue': 2}
SHAPES: tuple[str, ...] = ('square', 'circle')
PLURAL: dict[str, str] = {'square': 'squares', 'circle': 'circles'}
SIZES: tuple[int, ...] = (3, 4)
MIN_SHAPES, MAX_SHAPES = (1, 5)
VOCAB: tuple[str, ...] = ('<pad>', '<bos>', '<eos>', 'how', 'many', 'what', 'colour', 'is', 'the', 'largest', 'shape', 'shapes', 'red', 'green', 'blue', 'square', 'circle', 'squares', 'circles', 'observation', 'answer', ':', '0', '1', '2', '3', '4', '5')
STOI: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ITOS: tuple[str, ...] = VOCAB
VOCAB_SIZE = len(VOCAB)
PAD_ID = STOI['<pad>']
BOS_ID = STOI['<bos>']
EOS_ID = STOI['<eos>']
OBS_ID = STOI['observation']
ANSWER_ID = STOI['answer']
COLON_ID = STOI[':']
MAX_TEXT_LEN = 16

def encode(text: str) -> list[int]:
    """Whitespace tokenizer over the fixed vocabulary. ``"how many red squares" -> [3, 4, 12, 17]``."""
    raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

def decode(ids: list[int] | np.ndarray) -> str:
    """Inverse of :func:`encode`, dropping padding."""
    raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

@dataclass(frozen=True)
class Shape:
    """One coloured shape occupying one cell of the board."""
    colour: str
    kind: str
    row: int
    col: int
    size: int
    area: int

@dataclass
class Scene:
    """A list of shapes plus the image they render to."""
    shapes: list[Shape]
    image: np.ndarray = field(repr=False)

def _shape_mask(kind: str, size: int) -> np.ndarray:
    """Boolean stencil of one shape inside its ``size x size`` box. Returns (size, size)."""
    raise NotImplementedError('TODO: implement _shape_mask (see the reference in src/mlbook)')

def render(shapes: list[Shape], rng: np.random.Generator, noise: float=0.02) -> np.ndarray:
    """Draw shapes onto a black canvas. Returns (3, IMAGE_SIZE, IMAGE_SIZE) float32 in [0, 1]."""
    raise NotImplementedError('TODO: implement render (see the reference in src/mlbook)')

def sample_scene(rng: np.random.Generator) -> Scene:
    """Sample a board with a unique largest shape, then render it."""
    raise NotImplementedError('TODO: implement sample_scene (see the reference in src/mlbook)')

def count_shapes(scene: Scene, colour: str | None=None, kind: str | None=None) -> int:
    """Ground-truth count of shapes matching the filters. ``None`` means "any"."""
    raise NotImplementedError('TODO: implement count_shapes (see the reference in src/mlbook)')

def largest_colour(scene: Scene) -> str:
    """Colour of the shape with the most lit pixels. Ties are impossible by construction."""
    raise NotImplementedError('TODO: implement largest_colour (see the reference in src/mlbook)')

@dataclass
class Example:
    """One (image, question, answer) triple plus the scene that generated it."""
    image: np.ndarray
    question: str
    answer: str
    scene: Scene = field(repr=False)
    tool_query: tuple[str | None, str | None] | None = None
    'Arguments the ``count_shapes`` tool would need, or ``None`` if the tool\n    cannot answer this question. The tool has a *domain*; the agent loop has to\n    respect it.'

def sample_example(rng: np.random.Generator, scene: Scene | None=None) -> Example:
    """Sample a question about a (possibly supplied) scene and compute its answer."""
    raise NotImplementedError('TODO: implement sample_example (see the reference in src/mlbook)')

def make_dataset(n: int, seed: int=0) -> list[Example]:
    """``n`` independent (image, question, answer) triples from one seed."""
    raise NotImplementedError('TODO: implement make_dataset (see the reference in src/mlbook)')

def build_prompt(question: str, observation: str | None=None) -> list[int]:
    """Token ids of the prompt the model conditions on.

    Without a tool call:      ``<bos> how many red squares answer :``
    With a tool observation:  ``<bos> how many red squares observation : 2 answer :``

    The trailing ``answer :`` is the "assistant turn begins here" marker; the
    SFT loss is applied strictly after it (see :mod:`mlbook.capstone.sft_stage`).
    """
    raise NotImplementedError('TODO: implement build_prompt (see the reference in src/mlbook)')

def build_target(answer: str) -> list[int]:
    """Token ids of the assistant turn: the answer token followed by ``<eos>``."""
    raise NotImplementedError('TODO: implement build_target (see the reference in src/mlbook)')
