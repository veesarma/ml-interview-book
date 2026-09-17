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

The single fastest way to make this book worthless is for it to read as though a
machine produced it. A reader who spots the tells stops trusting the content.
`scripts/check_style.py` runs in CI and fails the build on the patterns below.
Run it on your own files before you finish:

```bash
python scripts/check_style.py docs/part05-sequence-transformers   # report
python scripts/check_style.py --stats                             # counts only
python scripts/fix_style.py docs/part05-sequence-transformers      # mechanical fixes
```

`fix_style.py` handles the mechanical cases (em dashes in headings, tables,
admonition titles and reference lists). Everything else you fix by rewriting the
sentence, because only the author knows what the sentence meant.

### Banned outright

**No em dashes.** Not one, anywhere, including inside headings, table cells,
admonition titles and reference lists. Use a comma, a colon, a full stop, or
parentheses. Em dashes are the loudest tell there is.

**No contrastive-binary template.** "It's not X, it's Y." "This isn't just a
detector, it's a perception system." "Not only fast but also accurate."
Interviews are about trade-offs, so you will constantly need contrast: state it
plainly instead. Write "Decode is bound by memory bandwidth, not compute," not
"Decode isn't about compute; it's about memory bandwidth."

**No false-candour filler.** "Let's be honest", "honestly", "to be fair",
"the honest answer is", "let's face it". It implies your other sentences were
less honest.

**No reveal scaffolding.** "Here's the thing", "here's the kicker", "but here's
where it gets interesting", "the real question is", "this is where X shines",
"Enter FlashAttention.", "let me walk you through", "let's unpack this",
"let's break it down", "think of it like".

**No rhetorical question you then answer.** "Why does this matter? Because..."
Delete the question and make the assertion.

**No summary-restating close.** "In summary", "In conclusion", "The takeaway",
"Bottom line", "At the end of the day". Stop on the last substantive sentence.

**No marketing vocabulary.** game-changer, paradigm shift, seamless, cutting-edge,
revolutionise, bulletproof, battle-tested, first-class citizen, under the hood,
unlock/unleash/harness the power of, cannot be overstated, double-edged sword,
silver bullet, tapestry, landscape of, realm of, the world of, journey through,
delve, dive into, in today's fast-paced world.

**No empty intensifiers.** crucial, pivotal, vital, essential, significantly,
dramatically, substantially, vastly, simply, trivially, obviously, clearly,
"it is easy to see". If something matters, say what breaks without it. If
something is faster, give the factor.

**No decorative emoji**, no ✅/🚀 bullets, no emoji in headings.

**No hedging tics.** "I could be wrong", "as an AI", "it's worth noting",
"it's important to note". State the claim, or state the uncertainty precisely
("the talk does not say whether they used X").

### Also avoid, though the linter cannot catch them

* **Forced triads.** Three parallel items because three sounds complete. Use two
  if there are two, four if there are four.
* **Epigram endings.** Landing every section on a compact quotable line.
* **Bold lead-ins on every bullet.** Vary the shape of your paragraphs and lists.
* **"That said" as a pivot** more than once per chapter.
* **Analogy inflation.** One good analogy per concept, introduced without fanfare.
* **Adjective stacking.** "clean, concise, and readable" is three words doing one
  word's work.
* **Uniform paragraph length.** Real writing has a two-word sentence next to a
  forty-word one.

### The test

Read a paragraph aloud. If it sounds like a conference keynote or a product
launch, rewrite it. If it sounds like a strong engineer explaining something at a
whiteboard to a colleague they respect, it is right. Aim for the register of
Karpathy's blog posts and OpenAI Spinning Up: direct, specific, unhurried,
willing to say "this is fiddly" or "nobody really knows why this works".

## 9. Do not

* Do not run `git` commands (the orchestrator commits).
* Do not add dependencies beyond numpy, scipy, torch, matplotlib, manim.
* Do not write "as we all know", "simply", "trivially", "it is easy to see".
* Do not leave TODOs, placeholders, or lorem ipsum. Ship complete chapters.
* Do not invent numbers (latencies, accuracies, company metrics), cite or omit.
