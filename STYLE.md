# Authoring guide (the contract every chapter follows)

This book has one reader: a senior/staff ML engineer preparing for interviews at
top companies (autonomy, consumer ranking, fintech, frontier labs). Every chapter
must let that reader move freely between

    equation to implementation to architecture to training dynamics to distributed system to evaluation to production case study.

The voice is the best of Karpathy (build it, show the shapes), Sebastian Raschka
(clean diagrams, honest trade-offs), OpenAI Spinning Up (derive it, then code it),
and Hello Interview (structure the answer, name the trade-off, commit to a decision).
Write in the second person, plainly, with no hand-waving. Production over research:
every technique is grounded in *when to use it, when not to, and who used it in production*.

---

## 1. Chapter skeleton (mandatory sections, in this order)

```markdown
# <Chapter title>

> **Why this matters at staff level.** 2–4 sentences: where this shows up in interviews
> (coding round, ML depth round, system design) and what "strong signal" looks like.

## TL;DR: the interview card
- 5–10 bullets. The equations, the shapes, the one-line trade-offs, the production users.
  This is what the reader re-reads the morning of the interview.

## 1. Intuition first
First principles. A concrete tiny example (numbers, a 2×3 matrix, a 4-token sequence).
Build the mental model *before* the general formula. Use a figure or mermaid diagram here.

## 2. The math
Derivations without gaps. Box the results the reader must reproduce at a whiteboard:
$$\boxed{\;\nabla_z L = p - y\;}$$
State assumptions. After each derivation, say in one sentence what it *means*.

## 3. Implementation
Code from `src/mlbook/...`, shown inline **with a shape comment on every tensor line**,
then explained paragraph by paragraph. Then a short "how you'd test it" block.
NumPy for foundations (Parts I–III, classical ML, optimizers), PyTorch where autograd,
batching or GPUs matter (Part IV onward). When both are instructive, show both under
`=== "NumPy"` / `=== "PyTorch"` content tabs.

## 4. Systems view: cost, failure modes, trade-offs
FLOPs, memory, latency, data requirements. When it works, when it breaks.
A **"when to use what"** table with a decision rule, not a feature list.

## 5. In production
2–5 real case studies. For each: company, what they built, *why they chose it*
(the trade-off), and a **link to the primary source** (engineering blog post, paper,
or talk). See §4 on links. Use the `production` admonition.

## 6. Interview questions and strong answers
5–10 questions, each with a model answer that shows *reasoning and trade-offs*
(not a list of facts). Include at least one "staff-level follow-up" per question.
Use the `interview` admonition.

## 7. Exercises
3–8 exercises graded ★ / ★★ / ★★★. Solutions inside collapsible `??? success "Solution"` blocks.
At least one exercise must be a coding exercise with a runnable check.

## References
Papers (with arXiv IDs), blog posts, talks. Every link verified (§4).
```

Chapter length: aim for 2,500–6,000 words of prose plus code. Depth beats breadth;
never pad. The reader should be able to *teach* the topic after reading.

## 2. Math conventions

* MathJax via `pymdownx.arithmatex` (generic). Inline `$...$`, display `$$...$$` on
  their own lines with a blank line before and after.
* Vectors lowercase bold-free ($x$), matrices uppercase ($W$), scalars greek/lowercase.
  Row-major data: $X \in \mathbb{R}^{N \times d}$, one example per row, so a linear
  layer is $XW + b$ (matches NumPy/PyTorch), **not** $Wx$. Say so when you deviate.
* Always give shapes next to symbols: "$Q \in \mathbb{R}^{T \times d_k}$".
* Available macros: `\R`, `\E`, `\KL`, `\softmax`, `\argmax`, `\argmin`, `\tr`, `\diag`, `\norm{x}`.
* Derive; do not cite a result you could derive in ten lines.

## 3. Code conventions (non-negotiable)

* **Explicit over clever.** Three separate `nn.Linear` for Q, K, V, never a fused
  `3*d` projection. No `einsum` unless the string is explained term-by-term in a
  comment. No `*args` magic, no metaprogramming, no monkey-patching.
* **Shape comment on every line that creates or reshapes a tensor**, in the form
  `# (B, T, d_model)`. Use a consistent dimension vocabulary:
  `B` batch, `T` sequence length, `d` / `d_model` model width, `H` heads, `d_head`,
  `V` vocab, `N` number of examples / tokens, `C,H,W` image channels/height/width,
  `K` classes, `E` experts, `L` layers.
* Functions ≤ ~40 lines; classes small; type hints; docstrings that state **input
  shapes, output shapes, and the equation implemented**.
* Pure NumPy in `mlbook/*` foundational modules (no scipy unless for a comparison
  test). PyTorch modules subclass `nn.Module`, take tensors, return tensors, no
  training loops inside model classes.
* Every implementation has a test in `tests/` that checks it against a known
  answer, a finite-difference gradient, or a reference PyTorch op (`torch.nn.functional`).
  Tests must run on CPU in < 10 s each. Use `torch.manual_seed` / `np.random.seed`.
* Package layout: `src/mlbook/<area>/<topic>.py` (see the area list in the task).
  Add each new module to the area's `__init__.py` `__all__`.
* Embed code in chapters either inline (preferred for explanation) or with the
  snippet directive so the book never drifts from the tested code:

  ```markdown
  ??? example "Full implementation, `src/mlbook/transformer/attention.py`"
      ```python
      --8<-- "src/mlbook/transformer/attention.py"
      ```
  ```

## 4. Links and case studies (verified, never invented)

* **Every external URL must be verified with the `WebSearch` tool before it goes in.**
  Direct page fetches are blocked in this environment; use search results to
  confirm the exact URL. If you cannot verify a URL, do not include it: give the
  company, the exact post/paper title, year, and venue instead (the reader can search).
* Prefer primary sources: the company's engineering blog, the arXiv abstract page
  (`https://arxiv.org/abs/<id>`), the official docs, or a recorded talk.
* Each case study answers: what was the business problem, what did they build,
  what alternative did they reject and why, what did it cost / gain.
* Never attribute a design choice to a company without a source. Mark inferences
  as inferences ("the public talk implies…").


### When your search budget runs out

Verifying links is the highest-value use of your WebSearch budget, because the
reader asked specifically for proof that a real company applied each technique.
Spend it on "In production" case studies and company claims first, and on generic
textbook references last.

If the budget runs out before you have verified everything, do not guess a URL.
Cite by title and venue as above, and append one line per unverified source to
`docs/_citations_todo/<your-part-dir>.md`:

```
- [ ] docs/part04-vision/04-detection.md | Focal Loss for Dense Object Detection | Lin et al., ICCV 2017 | believed arXiv:1708.02002
```

A later pass with fresh budget converts those into verified links. A source you
silently leave unrecorded never gets one.

## 5. Figures and animations

* Static figures: a matplotlib script in `figures/<part>_<name>.py` that writes
  `docs/assets/figures/<part>_<name>.png` (dpi=150, tight bbox, white background,
  colorblind-safe palette: use `plt.rcParams["axes.prop_cycle"]` default tab10).
  Run the script and commit the PNG. Reference as
  `![alt](../assets/figures/<part>_<name>.png){ width="640" }`.
* Manim scenes: `manim/scenes/<part>_<name>.py`. **Use `Text`, not `MathTex`/`Tex`
  (no LaTeX in the build environment).** Render a still with
  `manim -qm -s --format=png manim/scenes/<file>.py <SceneName>` and copy the PNG to
  `docs/assets/figures/`. If a GIF adds real value:
  `manim -ql --format=gif ...` and keep it under 2 MB.
* Mermaid for pipelines and data flow (renders natively):
  ```mermaid
  flowchart LR
    A[Image] --> B[Vision encoder] --> C[Projector] --> D[LLM]
  ```
* Every figure has a caption sentence in the prose explaining what to look at.

## 6. Admonitions available

`!!! note`, `!!! tip`, `!!! warning`, `!!! danger`, `!!! example`,
`!!! interview "Interview question"` (purple), `!!! production "Company, what they built"` (green),
collapsible variants with `???`. Content tabs: `=== "NumPy"` / `=== "PyTorch"`.

## 7. Cross-references

Link to other chapters with relative paths, e.g.
`[attention mathematics](../part05-sequence-transformers/03-attention-mathematics.md)`.
Never duplicate a derivation that lives elsewhere: link it and state the result.


## 8. Write like a person, not like a language model (enforced by a linter)

The fastest way to make this book worthless is for it to read as though a machine
produced it. A reader who spots the tells stops trusting the content.

The catalogue below is Ivo Velitchkov's "22 Claude-prose Patterns: A Catalog of
Stylistic Attractors in Generated Texts" (Link & Think, 25 June 2026), which
identifies the recurring rhetorical moves in generated prose. They are attractors:
states the writing keeps sliding back toward no matter what the prompt was. Knowing
the names is what lets you catch yourself.

`scripts/check_style.py` runs in CI and fails the build. Run it on your own files:

```bash
python scripts/check_style.py docs/part05-sequence-transformers
python scripts/check_style.py --stats
python scripts/fix_style.py docs/part05-sequence-transformers
```

Two severities. **ERROR** patterns fail on any occurrence. **DENSITY** patterns are
legitimate in technical prose but read as verbal reflexes when repeated, so they
fail only above a per-1000-word budget: one "in other words" is fine, nine are not.

### The 22 patterns

| Code | Pattern | What it looks like | Severity |
|---|---|---|---|
| SDA | Spaced-dash aside | Em dashes and spaced en dashes for asides and lists | error |
| CB | Contrastive binary | "not X but Y", "it is not A; it is B", "not just X, but Y" | error |
| CB2 | Bare contrastive binary | "X, not Y", "rather than" | density, 2 per 1k |
| SS | Significance-signaling | "this matters because", "which matters for" | error |
| AE | Aphoristic ender | Landing a section on a compact quotable epigram | density, 1 per 1k |
| MCS | Mirrored-clause symmetry | Two clauses in one frame, semicolon, slots swapped | density, 1.5 per 1k |
| MS | Meta-signposting | "below I", "as we saw above", "in what follows" | error |
| SRC | Self-ranking your claims | "the most important point", "the key insight is" | error |
| SH | Suspense hook | "has a name", "the cleanest idea is this" | error |
| SK | Stakes-raising | "shapes everything that follows", "the stakes are" | error |
| CDF | Candor flag | "the honest answer", "honestly", "to be fair" | error |
| VP | Validate then promise precision | "is correct, and it can be made precise" | error |
| RF | The reframe | "better posed:", "the better question is" | error |
| CP | Corrective pivot | "it would be wrong, though, to call it..." | error |
| ARR | Anticipate-and-rebut | "as though it carried no stance. It carries one." | error |
| CCC | Clean-consequence connector | "falls out of", "follows directly" | error |
| CL | Confidence by litotes | "not difficult", "not optional", "no accident" | error |
| DT | Deflating tail clause | "and no more", "and nothing else" | error |
| CF | Contribution framing | "supplies the other half", "fills the gap" | error |
| AHM | Reflexive AI-humility | "I could be wrong", "as a language model" | error |
| CR | Colon-reveal | "the answer is this:" setup, colon, payload | density, 0.5 per 1k |
| RG | Restatement gloss | "in other words", "put differently" | density, 1.5 per 1k |
| RH | Reflexive hedging | "tends to", "roughly", "largely", "by and large" | density, 4 per 1k |

Plus vocabulary rules: metadiscourse filler ("here's the thing", "let's unpack this",
"the real question is", "think of it like", "Enter FlashAttention."), overused
connectives (Moreover, Furthermore, Additionally, Thus, Notably, Crucially),
marketing words (game-changer, seamless, battle-tested, under the hood, delve,
tapestry, silver bullet, double-edged sword, unlock the power of), empty intensifiers
(crucial, pivotal, vital, simply, trivially, obviously, unquantified "significantly"),
lexical tics (genuinely, structurally, fundamentally, inherently), decorative emoji,
summary-restating closes, and rhetorical questions you then answer.

### Two notes on the hard cases

**Contrast is the content of this book.** Every chapter is about trade-offs, so you
will constantly need to say that A holds and B does not. The linter allows that as a
factual statement within a density budget (CB2) and rejects only the rhetorical
template (CB). Write "Decode is bound by memory bandwidth, not compute." Do not write
"Decode isn't about compute; it's about memory bandwidth."

**Significance is shown, not announced.** The mandated chapter opener
("Why this matters at staff level") is a structural affordance of a study book and is
exempt. In body prose, replace "This matters because the KV cache dominates memory at
long context" with "The KV cache dominates memory at long context." The consequence
is the point; the announcement is padding.

### Also avoid, though the linter cannot catch them

* **Forced triads.** Three parallel items because three sounds complete. Use two if
  there are two.
* **Bold lead-ins on every bullet.** Vary the shape of your paragraphs and lists.
* **Adjective stacking.** "clean, concise, and readable" is three words doing one
  word's work.
* **Uniform paragraph length.** Real writing has a two-word sentence next to a
  forty-word one.
* **Analogy inflation.** One analogy per concept, introduced without fanfare.
* **Pre-justification.** Explaining how to read a sentence before the sentence.
  It outsources the reader's orientation and forecloses their own reading.

### The test

Read a paragraph aloud. If it sounds like a conference keynote, rewrite it. If it
sounds like a strong engineer at a whiteboard explaining something to a colleague
they respect, it is right. Karpathy's blog posts and OpenAI Spinning Up are the
register: direct, specific, unhurried, willing to say "this is fiddly" or "nobody
really knows why this works".

## 9. Do not

* Do not run `git` commands (the orchestrator commits).
* Do not add dependencies beyond numpy, scipy, torch, matplotlib, manim.
* Do not write "as we all know", "simply", "trivially", "it is easy to see".
* Do not leave TODOs, placeholders, or lorem ipsum. Ship complete chapters.
* Do not invent numbers (latencies, accuracies, company metrics), cite or omit.
