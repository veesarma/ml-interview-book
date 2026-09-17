# LLMs and multimodal

> **Why this matters at staff level.** Everyone interviewing for these roles has read
> the same papers. The differentiator is being able to narrate how the field moved and
> why, with the pressure that forced each change named out loud: the KV cache drove the
> attention variants, training instability drove pre-norm and RMSNorm, PPO's
> operational pain drove DPO, and inference economics drove MoE and quantization. A
> candidate who can list the deltas from the 2017 Transformer and justify each one has
> demonstrated that they track the field instead of having memorised one paper.

## The questions

| # | Question | Taught properly in |
|---|---|---|
| 1 | [MHA to MQA to GQA to MLA](#q1) | [Efficient attention and the KV cache](../part06-llm-training/04-efficient-attention-kv-cache.md) |
| 2 | [Why RoPE, RMSNorm and SwiGLU became the standard recipe](#q2) | [Large-model architecture](../part06-llm-training/03-large-model-architecture.md) |
| 3 | [RLHF, then DPO, and why DPO emerged](#q3) | [DPO and its relatives](../part07-post-training/04-dpo-and-friends.md) |
| 4 | [LoRA and PEFT](#q4) | [Fine-tuning and LoRA](../part06-llm-training/06-fine-tuning-lora.md) |
| 5 | [RAG against fine-tuning, and RAG's failure modes](#q5) | [Retrieval and RAG](../part13-retrieval-eval-reliability/01-retrieval-and-rag.md) |
| 6 | [Mixture of Experts and its sharp edges](#q6) | [Large-model architecture](../part06-llm-training/03-large-model-architecture.md) |
| 7 | [Emergence, in-context learning, chain of thought, reasoning models](#q7) | [Scaling laws](../part06-llm-training/02-scaling-laws.md) |
| 8 | [CNNs to ViT to hybrids](#q8) | [Vision Transformers](../part08-multimodal/01-vision-transformers.md) |
| 9 | [Object detection's evolution and its design tensions](#q9) | [Object detection](../part04-vision/04-detection.md) |
| 10 | [Segmentation, and what SAM changed](#q10) | [Segmentation](../part04-vision/05-segmentation.md) |
| 11 | [CLIP, then LLaVA-style VLMs](#q11) | [VLM architecture](../part08-multimodal/04-vlm-architecture.md) |
| 12 | [Contrastive and self-supervised learning in vision](#q12) | [Self-supervised learning](../part10-self-supervised/01-self-supervised-learning.md) |

---

## The LLM stack

### 1. MHA to MQA to GQA to MLA {#q1}

??? question "Q: Trace the evolution of attention from multi-head to multi-query to grouped-query to multi-head latent attention."
    **The answer.** All four are responses to one pressure: the **KV cache dominates
    inference memory and bandwidth**, and vanilla multi-head attention stores a
    separate key and value per head.

    * **MHA**: $h$ query heads, $h$ key heads, $h$ value heads. Full expressivity, and
      the cache scales with $h$.
    * **MQA**: $h$ query heads sharing a **single** key and value head, shrinking the
      cache by a factor of $h$ and cutting the bandwidth the decode step needs. It was
      introduced for exactly this reason ([Shazeer,
      2019](https://arxiv.org/abs/1911.02150)), and it costs quality and some training
      stability, because every head reads from one key-value subspace.
    * **GQA**: the interpolation. Group the query heads into $g$ groups, each sharing
      one key-value pair. $g = h$ recovers MHA, $g = 1$ recovers MQA, and $g$ around 8
      gets most of MQA's memory saving at close to MHA quality ([Ainslie et al.,
      2023](https://arxiv.org/abs/2305.13245)). It is the modern default and a tunable
      knob.
    * **MLA**: instead of sharing, **compress**. Project keys and values into a
      low-rank latent that is what gets cached, and reconstruct per-head keys and
      values on the fly. DeepSeek-V2 reports a KV cache reduced by 93.3% against its
      predecessor alongside higher generation throughput ([DeepSeek-AI,
      2024](https://arxiv.org/abs/2405.04434)).

    !!! interview "Staff move"
        Name the single optimization target and connect it backwards. "The whole line
        optimises one thing, shrinking the KV cache to relieve the memory-bandwidth
        bottleneck of autoregressive decode, with progressively better tricks: MQA
        shares and loses quality, GQA groups and is the pragmatic sweet spot, MLA
        compresses into a latent and is the frontier move. It connects directly to the
        head-redundancy result: heads are redundant enough that sharing or compressing
        their keys and values barely hurts, which is why these schemes work at all. If
        asked what I would ship, GQA, because it is proven, simple, and has a dial. I
        would track MLA as its kernels mature, since the win is on the axis that
        binds."

    **Goes deeper:** [efficient attention and the KV cache](../part06-llm-training/04-efficient-attention-kv-cache.md)
    and [large-model architecture](../part06-llm-training/03-large-model-architecture.md).

### 2. Why RoPE, RMSNorm and SwiGLU became the standard recipe {#q2}

??? question "Q: Why did RoPE, RMSNorm and SwiGLU converge into the standard LLM recipe?"
    **The answer.** Each is the efficiency-and-stability-optimal choice in its slot,
    and they compose without interfering.

    * **RoPE** for position: relative-distance aware through rotation, parameter-free,
      defined at any length, and extrapolates better than a learned absolute table.
    * **RMSNorm** for normalization: LayerNorm minus the mean-centring, cheaper per
      token with no measured quality loss in Transformers.
    * **SwiGLU** for the FFN: a gated, multiplicative block that wins at matched FLOPs
      once the hidden width is scaled to two thirds.
    * **Pre-norm** for the residual topology: a clean identity path, which is what
      keeps multi-week runs from diverging.
    * **GQA** for the attention cache, from question 1.

    !!! interview "Staff move"
        Frame the answer as a diff. "The modern LLM is the 2017 Transformer with five
        well-motivated swaps: RoPE for sinusoidal absolute position, RMSNorm for
        LayerNorm, SwiGLU for the ReLU FFN, pre-norm for post-norm, GQA for full
        multi-head KV. None is individually revolutionary; together they are a
        meaningfully better-optimised machine, and each one was adopted because it
        improved the efficiency, stability or quality frontier without costing
        anything on the other two. Being able to list the deltas and justify each is
        the signal that you have tracked the field."

    **Goes deeper:** [large-model architecture](../part06-llm-training/03-large-model-architecture.md);
    the mechanisms are in [chapter 3, questions 7, 13, 14 and 15](03-deep-learning-transformers.md#q7).

### 3. RLHF, then DPO, and why DPO emerged {#q3}

??? question "Q: Explain RLHF, then DPO. Why did DPO emerge, and is it the end of the story?"
    **The answer.** A pretrained LLM predicts likely continuations, which is a
    different objective from being helpful, harmless and instruction-following.
    Alignment closes that gap.

    **RLHF**, the original three-stage pipeline:

    1. **SFT.** Fine-tune on high-quality demonstrations, an instruction paired with
       an ideal response.
    2. **Reward model.** Collect human preference comparisons between two responses to
       the same prompt and fit a reward model under a Bradley-Terry likelihood, so a
       reward difference maps to a preference probability.
    3. **RL optimization.** Optimise the policy to maximise the reward model's score
       with a **KL penalty** against the SFT reference, which is what stops the policy
       drifting into high-reward degenerate text.

    It works and it is operationally painful: a second model to train and maintain,
    PPO's hyperparameter sensitivity, careful KL control, reward hacking, and a
    sampling loop in the training path.

    **DPO** removes the reward model and the RL loop. The RLHF objective has a
    closed-form optimal policy, and reparameterising it lets the language model itself
    define the implicit reward, which collapses the problem into a single
    classification-style loss on preference pairs: raise the likelihood of the
    preferred response, lower the dispreferred, with the reference-model KL term baked
    into the loss ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). No
    reward model, no PPO, no sampling loop, a stable supervised-style update.

    !!! interview "Staff move"
        Give the trade-off and the fact that the story kept moving. "DPO won adoption
        because it is far simpler to implement and tune at comparable quality. The
        trade is that RLHF with a good reward model can still edge it out at the
        frontier, because online RL explores responses a fixed preference dataset does
        not contain, and the reward model generalises to states DPO never sees. There
        is now a zoo (IPO, KTO, online and iterative DPO), and RL came back decisively
        for reasoning through verifiable rewards and GRPO ([DeepSeek-AI,
        2025](https://arxiv.org/abs/2501.12948)). The framing I would use: DPO
        simplified preference alignment, and RL returned for reasoning, where the
        reward is checkable rather than learned. Those are different problems and they
        want different machinery."

    **Goes deeper:** [RLHF with PPO](../part07-post-training/03-rlhf-ppo.md),
    [reward models](../part07-post-training/02-reward-models.md),
    [DPO and its relatives](../part07-post-training/04-dpo-and-friends.md).

### 4. LoRA and PEFT {#q4}

??? question "Q: Explain LoRA and PEFT. Why does it work, and when does it not?"
    **The answer.** Full fine-tuning updates every parameter, which costs optimizer
    states for billions of weights and leaves you with a full model copy per task.
    Parameter-efficient fine-tuning updates a small set and freezes the rest.

    **LoRA** is the dominant method.

    * **Hypothesis.** The weight *update* during fine-tuning has low intrinsic rank, so
      approximate $\Delta W \in \R^{d \times k}$ as $BA$ with $B \in \R^{d \times r}$,
      $A \in \R^{r \times k}$, and $r \ll d, k$ (commonly 8 to 64).
    * **Mechanics.** Freeze $W$, compute $h = xW + xAB$ in row-major form, train only
      $A$ and $B$. $A$ is random, $B$ is zero, so the adapter contributes nothing at
      step zero and training starts from exactly the pretrained function. Trainable
      parameters fall by orders of magnitude, and the original paper reports 10,000x
      fewer trainable parameters and 3x lower GPU memory against Adam full fine-tuning
      of GPT-3 175B ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)).
    * **At inference.** Merge $BA$ into $W$ and serve one matrix, so there is **zero
      added latency**, unlike adapter layers that add sequential compute. Or keep many
      small adapters and swap them against one frozen base.
    * **QLoRA** adds a 4-bit quantized frozen base and paged optimizers, which is what
      made single-GPU fine-tuning of very large models routine ([Dettmers et al.,
      2023](https://arxiv.org/abs/2305.14314)).

    !!! note "Correction to a common phrasing"
        People say the zero-initialised $B$ makes the adapter "start as the identity".
        The adapter's **output** starts at zero, so the **layer** starts as the
        pretrained $W$. The distinction matters when you are reasoning about the
        gradient: at step zero, $A$ receives no gradient through $B = 0$, which is why
        $A$ is initialised randomly rather than also at zero, and why initialising both
        to zero leaves the adapter permanently dead.

    !!! interview "Staff move"
        Give the hypothesis, the operational payoff, and the boundary. "LoRA rests on
        fine-tuning moving the weights within a low-dimensional subspace, so a rank-$r$
        update suffices. The payoffs are operational: adapters of a few megabytes,
        mergeable for zero inference latency, and many task adapters served against
        one frozen base, which is a serving architecture and not just a training
        trick. The boundary is real: when the task needs large or genuinely high-rank
        weight change, such as a new domain or a new capability, LoRA underperforms
        full fine-tuning, and I would raise the rank or fine-tune fully. For style,
        instruction-following and task adaptation, LoRA is almost always the right
        call."

    **Goes deeper:** [fine-tuning and LoRA](../part06-llm-training/06-fine-tuning-lora.md).

### 5. RAG against fine-tuning, and RAG's failure modes {#q5}

??? question "Q: Explain RAG. When does it beat fine-tuning, and what are its failure modes?"
    **The answer.** Retrieval-augmented generation augments the model at inference:
    embed the query, retrieve the top-$k$ relevant chunks from a vector store or a
    hybrid keyword-plus-vector index, place them in the context, and generate grounded
    in them.

    **The decision rule is knowledge against behaviour.**

    * **RAG injects knowledge.** Use it when the model needs current, factual or
      proprietary information, especially when it changes. You update the index
      without retraining, you can cite sources, and grounding reduces hallucination.
    * **Fine-tuning changes behaviour.** Use it to teach tone, structure, output
      format, and domain reasoning patterns, which no amount of retrieved context
      reliably installs.
    * They compose. Fine-tune for the behaviour, retrieve for the knowledge.

    **Failure modes**, which is the staff-level half of the answer.

    * **Retrieval failure.** RAG quality is upper-bounded by retrieval quality.
      Embedding mismatch, chunking that splits a thought in half, and the
      vocabulary gap between query and document all show up as "the model
      hallucinated".
    * **Lost in the middle.** Models use the beginning and end of a long context far
      better than the middle, so a relevant chunk buried in position 7 of 15 can be
      ignored ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)).
    * **Conflicting or stale documents.** Retrieved context contradicts itself or the
      model's parametric knowledge, and nothing in the system decides which wins.
    * **Context dilution.** More chunks add noise and cost. Past a point, precision
      beats recall.
    * **The model ignores the context**, answering from parametric memory, or
      over-trusts a wrong retrieved passage.

    !!! interview "Staff move"
        Diagnose before prescribing, then design against the failure list. "The
        question I ask is whether the task fails for lack of knowledge or for lack of
        behaviour. Knowledge gaps point to retrieval, behaviour gaps to fine-tuning,
        and plenty of real systems need both. What juniors miss is that RAG's ceiling
        is the retriever, so most 'RAG is bad' situations are 'retrieval is bad', and
        I evaluate retrieval separately from generation with its own recall@k before I
        touch the prompt. I design for the failure modes explicitly: reranking so the
        best chunk lands where the model actually attends, provenance for auditability,
        and an abstention path so an empty retrieval produces 'I do not know' instead
        of fluent invention."

    **Goes deeper:** [retrieval and RAG](../part13-retrieval-eval-reliability/01-retrieval-and-rag.md).

### 6. Mixture of Experts and its sharp edges {#q6}

??? question "Q: What is a Mixture of Experts model, why use one, and what are the sharp edges?"
    **The answer.** MoE replaces the dense FFN with many parallel expert FFNs plus a
    **router** that selects a small subset per token, such as top-2 of 64. That
    decouples total parameters from per-token compute: enormous capacity, modest FLOPs
    per token. Switch Transformer took the routing to top-1 and reported up to 7x
    pre-training speedup at matched compute ([Fedus et al.,
    2021](https://arxiv.org/abs/2101.03961)), and the pattern now runs through frontier
    models, where DeepSeek-V2 activates 21B of 236B total parameters per token
    ([DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434)).

    **Sharp edges.**

    * **Load balancing.** The router collapses onto a few experts, leaving the rest
      untrained. An auxiliary load-balancing loss is what prevents it, and omitting it
      wastes most of your parameters.
    * **Memory.** Every expert is resident even though only a few run per token, so
      the memory footprint tracks total parameters while the FLOPs track active
      parameters.
    * **Training instability and non-differentiable routing.** Top-k selection is
      discrete, so you need noisy gating, auxiliary losses, and care with precision.
    * **Serving complexity.** Experts get sharded across devices, which is expert
      parallelism, and routing produces dynamic, uneven load and extra communication
      at inference.
    * **Batch-dependent behaviour.** Which experts fire depends on the batch
      composition, which complicates throughput planning and makes latency less
      predictable.

    !!! interview "Staff move"
        Reframe the trade in the axis the interviewer cares about. "MoE buys total
        capacity at fixed active compute, so it is a compute saver that you pay for in
        memory and systems complexity. That reframing is the whole answer: a team that
        is FLOP-bound in training should look at it, a team that is
        memory-capacity-bound at serving should be careful, and either way the routing
        and expert-parallel communication is real engineering, not a config flag. The
        load-balancing auxiliary loss is the detail to name unprompted, because
        without it the router collapses and you have paid for experts you never
        train."

    **Goes deeper:** [large-model architecture](../part06-llm-training/03-large-model-architecture.md).

### 7. Emergence, in-context learning, chain of thought, reasoning models {#q7}

??? question "Q: What do you make of emergent abilities, in-context learning, chain of thought, and the reasoning-model shift?"
    **The answer.** The capability story sitting on top of the architecture.

    * **In-context learning.** Large LMs perform tasks from examples in the prompt with
      no weight update. The mechanism is still debated, with induction heads and
      implicit optimization in the forward pass among the proposed accounts.
    * **Chain of thought.** Prompting the model to produce intermediate steps before
      the answer improves multi-step performance sharply, and the effect appears at
      sufficient scale ([Wei et al., 2022](https://arxiv.org/abs/2201.11903)). It
      gives the model serial compute and conditions generation on its own intermediate
      work.
    * **Emergent abilities.** Capabilities that appear past a scale threshold. Carry
      the live counter-argument: some apparent emergence is an artifact of
      discontinuous metrics, where a smoothly improving capability looks like a phase
      transition because the metric is thresholded ([Schaeffer et al.,
      2023](https://arxiv.org/abs/2304.15004)).
    * **The reasoning shift.** Training with RL against verifiable rewards (does the
      code pass, is the proof correct) to produce long chains of thought before
      answering, which is what DeepSeek-R1 demonstrated with GRPO and pure RL from a
      base model ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)). It added
      **inference-time compute** as a scaling axis alongside parameters and data.

    !!! note "A note on dating"
        Source material written in 2024 and 2025 describes the reasoning shift as new.
        Treat it as the established third scaling axis: spending more tokens at
        inference is now a deliberate product and cost decision, with model families
        exposing an explicit reasoning-effort control. The interview-relevant content
        is no longer "this exists" but "what does it cost and when is it worth it",
        which is a latency and unit-economics question.

    !!! interview "Staff move"
        Narrate the arc and keep the skepticism. "Scale gave in-context learning and
        chain of thought without anyone designing them, and then the frontier added an
        axis: trade more thinking tokens at inference for better answers, trained via
        RL where the reward is checkable. That is why RL returned after DPO appeared
        to have simplified alignment, because preference alignment and reasoning are
        different problems with different reward structures. I would also flag the
        emergence-as-metric-artifact result, because treating every benchmark
        step-change as a capability phase transition is how teams end up planning
        around a threshold that does not exist."

    **Goes deeper:** [scaling laws](../part06-llm-training/02-scaling-laws.md),
    [reasoning RL, RLVR and GRPO](../part07-post-training/05-reasoning-rl-grpo.md),
    [test-time compute](../part07-post-training/06-test-time-compute.md).

---

## Vision and multimodal

### 8. CNNs to ViT to hybrids {#q8}

??? question "Q: How has computer-vision architecture evolved from CNNs to ViT to hybrids?"
    **The answer.** The arc is inductive bias against scale, the same theme as
    attention against convolution.

    **The CNN era.** Convolution encodes two priors that match natural images:
    **locality** and **translation equivariance**, through weight sharing. Stacked
    convolutions compose hierarchically, edges to textures to parts to objects, with
    growing receptive fields and spatial downsampling for invariance and efficiency.
    The milestones: AlexNet (deep CNNs plus GPUs plus ImageNet), VGG (depth with small
    3x3 filters), **ResNet** (residual connections solve the degradation problem by
    giving gradients an identity path, which is the same idea that powers pre-norm
    Transformers), then the efficiency line of MobileNet's depthwise-separable
    convolutions and EfficientNet's compound scaling.

    **ViT.** Split the image into patches, embed each linearly as a token, add
    positional embeddings, run a standard Transformer encoder, use no convolution
    ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)). Dropping the
    locality prior means attention can relate any two patches at $O(1)$ path length,
    and it means the model must learn structure the CNN assumed. The consequence is
    the trade-off worth naming: ViTs underperform CNNs on small and medium datasets
    and match or beat them with enough data or large-scale pretraining.

    **Convergence.** **Swin** reintroduced locality and hierarchy through shifted
    window attention, giving linear complexity in image size and multi-scale features,
    which made Transformers practical for detection and segmentation ([Liu et al.,
    2021](https://arxiv.org/abs/2103.14030)). **ConvNeXt** then modernised a pure CNN
    with Transformer-era training recipes and design choices and matched ViTs
    ([Liu et al., 2022](https://arxiv.org/abs/2201.03545)), which showed how much of
    the gap had been training methodology.

    !!! interview "Staff move"
        Refuse architecture tribalism and name the confound. "CNNs encode priors that
        are a free lunch when data is limited and a ceiling when it is abundant; ViTs
        drop them for generality and win at scale; ConvNeXt showed a chunk of ViT's
        advantage was the modern recipe, augmentation, AdamW, long schedules, large
        data, and not self-attention. So for a data-constrained on-device perception
        problem I might still pick an efficient CNN, and for a pretraining-rich setting
        a Transformer or hybrid, and either way I insist on matched training recipes
        before believing an architecture comparison, because that confound has misled a
        lot of published work. ResNet's identity path is the through-idea linking CNNs
        and Transformers."

    **Goes deeper:** [CNN architectures](../part04-vision/03-cnn-architectures.md) and
    [Vision Transformers](../part08-multimodal/01-vision-transformers.md).

### 9. Object detection's evolution and its design tensions {#q9}

??? question "Q: Walk me through object detection's evolution and the design tensions that persist."
    **The answer.** The lineage, organised by the core problem of localising and
    classifying a variable number of objects.

    * **Two-stage** (R-CNN, Fast, Faster R-CNN): propose candidate regions with a
      learned region proposal network, then classify and refine each. Accurate,
      slower. Contributed RoIPool and RoIAlign and anchor boxes.
    * **One-stage** (YOLO, SSD, RetinaNet): predict boxes and classes densely over a
      grid in one pass. Faster, historically less accurate, until **focal loss** closed
      much of the gap by fixing the extreme foreground-background imbalance: a dense
      detector sees on the order of 100k easy background anchors per image, and
      down-weighting them by $(1-p_t)^\gamma$ lets the rare positives drive the
      gradient ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)).
    * **Anchor-free** (FCOS, CenterNet): drop hand-designed anchor boxes, which carry
      hyperparameters and imbalance of their own, and predict centres plus sizes.
    * **DETR**: reframe detection as **set prediction**. A Transformer with learned
      object queries outputs a set of boxes, trained by bipartite Hungarian matching
      against ground truth, which removes anchors and non-maximum suppression entirely
      and makes detection end-to-end ([Carion et al.,
      2020](https://arxiv.org/abs/2005.12872)). It paid in slow convergence, which
      Deformable DETR and DINO later addressed.

    **The persistent tensions**: accuracy against latency (two-stage against
    one-stage), hand-designed priors against learned ones (anchors and NMS against set
    prediction), and the **NMS problem** itself, a non-differentiable,
    threshold-sensitive post-process that breaks end-to-end training and degrades in
    crowded scenes where true objects overlap.

    !!! interview "Staff move"
        Pull out two themes and then commit. "First, focal loss is the canonical case
        of solving a problem with a loss instead of an architecture: the blocker for
        one-stage detectors was an imbalance, and reweighting the loss fixed it.
        Second, the field's trajectory is removing hand-engineered components, anchors
        then NMS, in favour of learned end-to-end formulations. For a production choice
        I anchor on the latency budget: real-time on-device gets an efficient
        anchor-free one-stage model, accuracy-critical offline gets a DETR-family or
        two-stage model, and I flag NMS as a deployment sharp edge wherever the scenes
        are crowded, because that is where the threshold you tuned on a benchmark stops
        holding."

    **Goes deeper:** [object detection](../part04-vision/04-detection.md) and
    [DETR and set prediction](../part08-multimodal/02-detr.md).

### 10. Segmentation, and what SAM changed {#q10}

??? question "Q: How does segmentation work, and what changed with foundation models like SAM?"
    **The answer.** Segmentation is per-pixel labelling in three flavours:
    **semantic** (every pixel gets a class, no instances), **instance** (separate
    object instances, as in Mask R-CNN, which is Faster R-CNN plus a per-region mask
    head), and **panoptic** (both at once, every pixel labelled and every thing
    instanced).

    Architecturally the recurring pattern is **encoder-decoder with skip connections**.
    The encoder downsamples to capture semantics, the decoder upsamples back to full
    resolution, and U-Net skip connections carry high-resolution spatial detail across
    so that boundaries stay crisp. Dilated convolutions (DeepLab) enlarge the receptive
    field without giving up resolution.

    **SAM** made segmentation promptable. Trained on a dataset of over 1 billion masks
    built with a model-in-the-loop data engine, it takes a point, box or mask prompt
    and segments the indicated object zero-shot, transferring to image distributions
    and tasks it never trained on ([Kirillov et al.,
    2023](https://arxiv.org/abs/2304.02643)). That moved segmentation from "train a
    bespoke model per dataset and class list" toward "prompt a general model", and
    follow-ons extend it to video with a memory mechanism.

    !!! interview "Staff move"
        Name the constant, then the shift, then the deployment caveat. "The
        architectural constant is encoder-decoder with skips, because you need both
        deep semantics and precise boundaries and the skips are what carry the second.
        The shift is that segmentation became a promptable zero-shot problem, which
        changes the practice: instead of collecting and labelling a dataset per task, I
        prompt or fine-tune a foundation model and spend the labelling budget on the
        hard residual. SAM's own training used a data engine, which is the auto-labeling
        story from the other direction. For deployment I would still distil down to a
        task-specific efficient model for on-device, because a SAM-scale encoder does
        not fit a real-time edge budget."

    **Goes deeper:** [segmentation](../part04-vision/05-segmentation.md) and
    [perception foundation models](../part11-perception-autonomy/01-perception-foundation-models.md).

### 11. CLIP, then LLaVA-style VLMs {#q11}

??? question "Q: How do vision-language models work? Contrast CLIP with a LLaVA-style VLM."
    **The answer.** Two paradigms worth separating cleanly.

    **CLIP, a contrastive dual encoder.** Train an image encoder and a text encoder
    jointly on hundreds of millions of image-caption pairs with an InfoNCE objective:
    within a batch, pull each image embedding toward its caption and push it away from
    every other caption, as a symmetric cross-entropy over the similarity matrix. The
    result is a shared space where images and text are comparable, which gives
    **zero-shot classification**: embed the image, embed the candidate class names as
    text prompts, take the nearest ([Radford et al.,
    2021](https://arxiv.org/abs/2103.00020)). CLIP is the backbone of open-vocabulary
    detection and segmentation and the text conditioning in text-to-image models.

    **LLaVA-style VLMs, generative and LLM-centric.** Connect a vision encoder, often
    a frozen CLIP image tower, to an LLM through a **projector**: an MLP, or a
    cross-attention resampler in the style of Flamingo's Perceiver or BLIP-2's
    Q-Former. The projector maps image features into the LLM's token embedding space,
    so the image becomes a short sequence of visual tokens the LLM consumes alongside
    text. Training is usually two stages: align the projector on image-text pairs with
    the encoder and often the LLM frozen, then instruction-tune on visual instruction
    data ([Liu et al., 2023](https://arxiv.org/abs/2304.08485)). That yields VQA,
    captioning, visual reasoning, grounding and document understanding.

    **The contrast.** CLIP is discriminative: a shared space for matching, fixed-length
    embeddings, excellent for retrieval and zero-shot classification, and it cannot
    generate text. A VLM is generative: free-form text about images, reasoning and
    dialogue, at higher cost and with hallucination risk. They are not rivals, because
    the VLM usually has a CLIP-trained encoder inside it.

    !!! interview "Staff move"
        Name the design decisions a VLM actually forces. "Discriminative alignment
        against generative fusion. CLIP aligns two modalities with a contrastive loss,
        which is cheap and scales on noisy web pairs, and it is the engine behind
        zero-shot and open-vocabulary vision. A LLaVA-style model fuses a vision
        encoder into the LLM's token space and generates. The engineering decisions
        that matter are the connector (a simple MLP against a resampler, which is a
        compute and quality trade), how many visual tokens you feed, which sets
        resolution against context cost, and what you freeze at each stage. And the
        efficiency theme returns: visual tokens inflate the sequence, so
        high-resolution understanding runs straight into the $O(n^2)$ attention cost,
        which is why token reduction and resampler design are where the work is."

    **Goes deeper:** [CLIP and contrastive learning](../part08-multimodal/03-clip-contrastive.md)
    and [VLM architecture](../part08-multimodal/04-vlm-architecture.md).

### 12. Contrastive and self-supervised learning in vision {#q12}

??? question "Q: What are the big ideas in contrastive and self-supervised learning for vision?"
    **The answer.** Labels are expensive and images are abundant, so learn
    representations from unlabelled images with a pretext task and fine-tune on a small
    labelled set.

    **Contrastive (SimCLR, MoCo).** Build two augmented views of the same image as a
    positive pair, treat other images as negatives, and learn an embedding where
    positives are close and negatives far under InfoNCE. The augmentations *are* the
    supervision: cropping and colour jitter declare that position and colour should not
    change the representation, and SimCLR showed the augmentation composition is what
    determines the quality of the learned features ([Chen et al.,
    2020](https://arxiv.org/abs/2002.05709)). MoCo contributed a momentum encoder and a
    queue of negatives, which decouples the number of negatives from the batch size.
    A sharp edge worth naming: with BatchNorm in the encoder, information leaks between
    examples through the batch statistics, so the model can identify the positive from
    batch structure instead of content, which MoCo addressed with shuffling BN. That is
    the same "BN assumes IID batches" lesson from
    [chapter 3, question 7](03-deep-learning-transformers.md#q7), surfacing in a new
    place.

    **Non-contrastive and masked modelling.** BYOL and DINO drop negatives entirely: a
    student predicts a momentum teacher's representation and avoids collapse through
    architectural asymmetry and a stop-gradient. DINO's attention maps show emergent
    object segmentation with no segmentation labels. **MAE** masks about 75% of the
    patches and reconstructs them, which is the BERT idea for vision and scales
    cleanly, with the encoder seeing only the visible patches ([He et al.,
    2021](https://arxiv.org/abs/2111.06377)).

    !!! interview "Staff move"
        Name the design decision that is actually being made. "The unifying idea is
        that augmentation or masking defines the supervision. You are telling the model
        'these two things should mean the same' or 'reconstruct the hidden part', so
        the choice of augmentation or mask ratio is the real design decision, because
        it determines which invariances the representation learns. If I need an
        encoder that keeps colour information, colour jitter is the wrong augmentation
        no matter what the benchmark says. I would also connect the contrastive
        BatchNorm leak to the general BN-and-IID lesson, because recognising the same
        root cause across contexts is the kind of transfer the round is looking for."

    **Goes deeper:** [self-supervised learning](../part10-self-supervised/01-self-supervised-learning.md)
    and [CLIP and contrastive learning](../part08-multimodal/03-clip-contrastive.md).

## References

* Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need", 2019.
  [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
* Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints", EMNLP 2023.
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
* DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
  Language Model", 2024. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
* Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a
  Reward Model", NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
* DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning", 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
* Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
* Dettmers, Pagnoni, Holtzman, Zettlemoyer. "QLoRA: Efficient Finetuning of Quantized
  LLMs", NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
* Liu et al. "Lost in the Middle: How Language Models Use Long Contexts", TACL 2024.
  [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
* Fedus, Zoph, Shazeer. "Switch Transformers", JMLR 2022.
  [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
* Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
  NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
* Schaeffer, Miranda, Koyejo. "Are Emergent Abilities of Large Language Models a
  Mirage?", NeurIPS 2023. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
* Dosovitskiy et al. "An Image is Worth 16x16 Words", ICLR 2021.
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
* Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
  ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
* Liu, Mao, Wu, Feichtenhofer, Darrell, Xie. "A ConvNet for the 2020s", CVPR 2022.
  [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
* Lin, Goyal, Girshick, He, Dollár. "Focal Loss for Dense Object Detection", ICCV 2017.
  [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
* Carion et al. "End-to-End Object Detection with Transformers", ECCV 2020.
  [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
* Kirillov et al. "Segment Anything", ICCV 2023.
  [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
* Radford et al. "Learning Transferable Visual Models From Natural Language
  Supervision", ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
* Liu, Li, Wu, Lee. "Visual Instruction Tuning", NeurIPS 2023.
  [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
* Chen, Kornblith, Norouzi, Hinton. "A Simple Framework for Contrastive Learning of
  Visual Representations", ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
* He, Chen, Xie, Li, Dollár, Girshick. "Masked Autoencoders Are Scalable Vision
  Learners", CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
