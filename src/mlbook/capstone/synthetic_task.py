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

# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

GRID = 4                        # cells per side
CELL = 4                        # pixels per cell (also the ViT patch size)
IMAGE_SIZE = GRID * CELL        # 16 pixels per side
N_CELLS = GRID * GRID           # 16 cells == 16 patches

COLOURS: tuple[str, ...] = ("red", "green", "blue")
CHANNEL: dict[str, int] = {"red": 0, "green": 1, "blue": 2}
SHAPES: tuple[str, ...] = ("square", "circle")
PLURAL: dict[str, str] = {"square": "squares", "circle": "circles"}
SIZES: tuple[int, ...] = (3, 4)  # side length in pixels, inside a 4x4 cell
MIN_SHAPES, MAX_SHAPES = 1, 5    # so every count answer lies in 0..5

# ---------------------------------------------------------------------------
# The vocabulary and its whitespace tokenizer
# ---------------------------------------------------------------------------

VOCAB: tuple[str, ...] = (
    "<pad>", "<bos>", "<eos>",
    "how", "many", "what", "colour", "is", "the", "largest", "shape", "shapes",
    "red", "green", "blue",
    "square", "circle", "squares", "circles",
    "observation", "answer", ":",
    "0", "1", "2", "3", "4", "5",
)
STOI: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ITOS: tuple[str, ...] = VOCAB
VOCAB_SIZE = len(VOCAB)

PAD_ID = STOI["<pad>"]
BOS_ID = STOI["<bos>"]
EOS_ID = STOI["<eos>"]
OBS_ID = STOI["observation"]
ANSWER_ID = STOI["answer"]
COLON_ID = STOI[":"]

MAX_TEXT_LEN = 16               # longest prompt + answer + <eos>, see build_prompt


def encode(text: str) -> list[int]:
    """Whitespace tokenizer over the fixed vocabulary. ``"how many red squares" -> [3, 4, 12, 17]``."""
    return [STOI[word] for word in text.split()]


def decode(ids: list[int] | np.ndarray) -> str:
    """Inverse of :func:`encode`, dropping padding."""
    return " ".join(ITOS[int(i)] for i in ids if int(i) != PAD_ID)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """One coloured shape occupying one cell of the board."""

    colour: str
    kind: str          # "square" or "circle"
    row: int           # cell row, 0..GRID-1
    col: int           # cell column, 0..GRID-1
    size: int          # side length in pixels, in SIZES
    area: int          # number of lit pixels; the ground truth for "largest"


@dataclass
class Scene:
    """A list of shapes plus the image they render to."""

    shapes: list[Shape]
    image: np.ndarray = field(repr=False)  # (3, IMAGE_SIZE, IMAGE_SIZE) float32


def _shape_mask(kind: str, size: int) -> np.ndarray:
    """Boolean stencil of one shape inside its ``size x size`` box. Returns (size, size)."""
    mask = np.ones((size, size), dtype=bool)  # (size, size)
    if kind == "circle":
        # A "circle" is the box with its four corner pixels knocked off, which
        # gives areas 5 (size 3) and 12 (size 4) -- distinct from the squares'
        # 9 and 16, so "which shape is largest" is never a tie between kinds.
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    return mask


def render(shapes: list[Shape], rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    """Draw shapes onto a black canvas. Returns (3, IMAGE_SIZE, IMAGE_SIZE) float32 in [0, 1]."""
    image = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)  # (3, 16, 16)
    for shape in shapes:
        stencil = _shape_mask(shape.kind, shape.size)                # (s, s)
        pad = (CELL - shape.size) // 2
        y0 = shape.row * CELL + pad
        x0 = shape.col * CELL + pad
        patch = image[CHANNEL[shape.colour], y0:y0 + shape.size, x0:x0 + shape.size]  # (s, s)
        patch[stencil] = 1.0
    image += noise * rng.standard_normal(image.shape).astype(np.float32)  # (3, 16, 16)
    return np.clip(image, 0.0, 1.0)


def sample_scene(rng: np.random.Generator) -> Scene:
    """Sample a board with a unique largest shape, then render it."""
    while True:
        n = int(rng.integers(MIN_SHAPES, MAX_SHAPES + 1))
        cells = rng.choice(N_CELLS, size=n, replace=False)           # (n,) flat cell indices
        shapes: list[Shape] = []
        for cell in cells:
            colour = COLOURS[int(rng.integers(len(COLOURS)))]
            kind = SHAPES[int(rng.integers(len(SHAPES)))]
            size = SIZES[int(rng.integers(len(SIZES)))]
            area = int(_shape_mask(kind, size).sum())
            shapes.append(Shape(colour, kind, int(cell) // GRID, int(cell) % GRID, size, area))
        areas = [s.area for s in shapes]
        if areas.count(max(areas)) == 1:      # reject ties so "largest" is well defined
            return Scene(shapes=shapes, image=render(shapes, rng))


# ---------------------------------------------------------------------------
# The programmatic oracle: every answer is computed, never labelled
# ---------------------------------------------------------------------------


def count_shapes(scene: Scene, colour: str | None = None, kind: str | None = None) -> int:
    """Ground-truth count of shapes matching the filters. ``None`` means "any"."""
    return sum(
        1
        for s in scene.shapes
        if (colour is None or s.colour == colour) and (kind is None or s.kind == kind)
    )


def largest_colour(scene: Scene) -> str:
    """Colour of the shape with the most lit pixels. Ties are impossible by construction."""
    return max(scene.shapes, key=lambda s: s.area).colour


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """One (image, question, answer) triple plus the scene that generated it."""

    image: np.ndarray            # (3, IMAGE_SIZE, IMAGE_SIZE) float32
    question: str                # e.g. "how many red squares"
    answer: str                  # one vocabulary token: "0".."5" or a colour
    scene: Scene = field(repr=False)
    tool_query: tuple[str | None, str | None] | None = None
    """Arguments the ``count_shapes`` tool would need, or ``None`` if the tool
    cannot answer this question. The tool has a *domain*; the agent loop has to
    respect it."""


def sample_example(rng: np.random.Generator, scene: Scene | None = None) -> Example:
    """Sample a question about a (possibly supplied) scene and compute its answer."""
    scene = sample_scene(rng) if scene is None else scene
    template = int(rng.integers(4))
    if template == 0:                                   # "how many red squares"
        colour = COLOURS[int(rng.integers(len(COLOURS)))]
        kind = SHAPES[int(rng.integers(len(SHAPES)))]
        question = f"how many {colour} {PLURAL[kind]}"
        answer = str(count_shapes(scene, colour, kind))
        tool_query: tuple[str | None, str | None] | None = (colour, kind)
    elif template == 1:                                 # "how many red shapes"
        colour = COLOURS[int(rng.integers(len(COLOURS)))]
        question = f"how many {colour} shapes"
        answer = str(count_shapes(scene, colour, None))
        tool_query = (colour, None)
    elif template == 2:                                 # "how many shapes"
        question = "how many shapes"
        answer = str(count_shapes(scene))
        tool_query = (None, None)
    else:                                               # "what colour is the largest shape"
        question = "what colour is the largest shape"
        answer = largest_colour(scene)
        tool_query = None
    return Example(image=scene.image, question=question, answer=answer, scene=scene, tool_query=tool_query)


def make_dataset(n: int, seed: int = 0) -> list[Example]:
    """``n`` independent (image, question, answer) triples from one seed."""
    rng = np.random.default_rng(seed)
    return [sample_example(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(question: str, observation: str | None = None) -> list[int]:
    """Token ids of the prompt the model conditions on.

    Without a tool call:      ``<bos> how many red squares answer :``
    With a tool observation:  ``<bos> how many red squares observation : 2 answer :``

    The trailing ``answer :`` is the "assistant turn begins here" marker; the
    SFT loss is applied strictly after it (see :mod:`mlbook.capstone.sft_stage`).
    """
    ids = [BOS_ID] + encode(question)
    if observation is not None:
        ids += [OBS_ID, COLON_ID] + encode(observation)
    ids += [ANSWER_ID, COLON_ID]
    return ids


def build_target(answer: str) -> list[int]:
    """Token ids of the assistant turn: the answer token followed by ``<eos>``."""
    return encode(answer) + [EOS_ID]
