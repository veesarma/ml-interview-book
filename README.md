# The ML Interview Book

**From first principles to the frontier, for staff-level ML interviews.**

A book-format compendium (MkDocs Material, deployable to GitHub Pages) covering the
mathematics, from-scratch implementations, systems trade-offs, and production case
studies behind modern machine learning, with dedicated parts on ML system design
and company-specific deep dives.

> Read it at **https://veesarma.github.io/ml-interview-book/** once GitHub Pages is
> enabled for the repository (Settings → Pages → Source: `gh-pages` branch; the
> workflow in `.github/workflows/deploy.yml` publishes on every push to `main`).

## What's inside

| Part | Theme |
|---|---|
| I–III | Math, classical ML, neural nets from scratch (NumPy autograd engine) |
| IV | Computer vision: convolutions, CNNs, detection, segmentation, geometry, 3D |
| V | Sequence models and the Transformer (attention, GPT, KV cache, RoPE, BPE) |
| VI | LLM training: data, scaling laws, MoE/GQA/SSMs, FlashAttention, quantization, LoRA, mid-training |
| VII | Post-training: SFT, reward models, RLHF/PPO, DPO, GRPO/RLVR, test-time compute |
| VIII | ViT, DETR, CLIP, VLM architecture, multimodal foundation and video models |
| IX | Generative models: VAE, GAN, diffusion, flow matching |
| X | Self-, semi- and weakly supervised learning and auto-labelling in production |
| XI | Perception & autonomy: foundation models, BEV, fusion, tracking, occupancy, prediction/planning, world models |
| XII | Reinforcement learning from the MDP to PPO, imitation learning, agents |
| XIII | Retrieval & RAG, evaluation, uncertainty & reliability |
| XIV | Distributed training, training systems, inference systems, hardware & roofline |
| XV | Interpretability and safety / failure modes |
| XVI | The coding canon: 60 implementations, NumPy ↔ PyTorch fluency drills |
| XVII | ML system design: framework + 12 canonical designs grounded in company write-ups |
| XVIII | Company deep dives: Tesla, Waymo, Zoox/Nuro/Aurora, NVIDIA, Meta, Google/YouTube, TikTok, Netflix/Spotify, Pinterest, Uber/DoorDash, Airbnb, Amazon, Apple, Stripe/fintech, frontier labs, Scale AI |

Every chapter follows the same skeleton: an interview card, intuition, derivations,
an explicit implementation with shapes on every line, a systems view, production
case studies with primary-source links, interview questions with strong answers,
and exercises with solutions.

## Layout

```
docs/               the book (Markdown, MathJax, Mermaid)
src/mlbook/         from-scratch implementations, one area per package
tests/              every implementation is tested against a reference
figures/            matplotlib scripts that generate docs/assets/figures/*.png
manim/scenes/       Manim scenes for animated/still explanatory figures
STYLE.md            the authoring contract every chapter follows
scripts/            style linter and practice-stub generator
practice/           signature-only stubs for retyping the canon by hand
snippets/           the generated bibliography included by docs/references.md
```

## Run it locally

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q            # run all implementation tests
mkdocs serve                   # http://127.0.0.1:8000
```

### Retype the canon by hand

```bash
make stubs                              # signature-only stubs of every module
make drill ITEM=transformer/multihead   # shows the file to edit and the tests to pass
#   ... write practice/transformer/multihead.py from memory ...
make check ITEM=transformer/multihead   # grades your version against the real tests
make reset ITEM=transformer/multihead   # start over from a clean stub
```

Check the prose against the AI-tell linter with `make style` (it runs in CI and
enforces STYLE.md section 8). Regenerate figures with `make figures`; render a Manim still with
`manim -qm -s --format=png manim/scenes/<file>.py <SceneName>`.
