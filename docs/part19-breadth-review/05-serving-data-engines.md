# Serving, data engines and the real world

> **Why this matters at staff level.** These questions decide whether an interviewer
> believes you have run something in production. The serving half has one governing
> idea: inference optimization is a latency, throughput, cost and quality
> quadrilateral, and you cannot maximise all four, so a strong answer names the axis
> it is spending. The data half has another: at scale the bottleneck is labelled data
> and not modelling, so the deliverable is the loop that produces data, and the model
> is an artifact of the loop.

## The questions

| # | Question | Taught properly in |
|---|---|---|
| 1 | [How to reduce inference latency](#q1) | [Inference systems](../part14-systems/03-inference-systems.md) |
| 2 | [How to increase throughput](#q2) | [Inference systems](../part14-systems/03-inference-systems.md) |
| 3 | [Caching trade-offs](#q3) | [Efficient attention and the KV cache](../part06-llm-training/04-efficient-attention-kv-cache.md) |
| 4 | [What a staff engineer should know about large-scale training](#q4) | [Distributed training](../part14-systems/01-distributed-training.md) |
| 5 | [Auto-labeling and building a data engine](#q5) | [Weak supervision and auto-labeling](../part10-self-supervised/03-weak-supervision-and-auto-labeling.md) |
| 6 | [Choosing which examples to label](#q6) | [Semi-supervised learning](../part10-self-supervised/02-semi-supervised.md) |
| 7 | [The long tail, edge cases and shift](#q7) | [AV perception system design](../part17-ml-system-design/05-perception-system-av.md) |
| 8 | [Sensor fusion and multimodal perception](#q8) | [Sensor fusion](../part11-perception-autonomy/03-sensor-fusion.md) |

---

## Serving and training systems

### 1. How to reduce inference latency {#q1}

??? question "Q: How do you reduce inference latency?"
    **The answer.** Latency is per-request wall clock. Work from the model down to the
    system.

    **Model level.**

    * **Quantization**, fp32 to fp16 or bf16 to int8 to int4. Fewer bits means less
      memory traffic, which is usually the real bottleneck, and faster math where the
      hardware supports it. Post-training quantization is cheap; quantization-aware
      training recovers the accuracy you lost. This is the dominant on-device lever.
    * **Pruning.** Structured pruning removes whole channels or heads and speeds up
      dense hardware immediately. Unstructured pruning needs sparse kernels before it
      pays anything.
    * **Distillation.** Train a small student against a large teacher's soft outputs
      and ship the student.
    * **Architecture.** Depthwise-separable convolutions, efficient attention,
      early-exit networks.

    **Compiler and kernel level.** Operator fusion and graph optimization (TensorRT,
    ONNX Runtime, XLA, `torch.compile`) collapse conv-bn-relu into one kernel and
    delete redundant ops. Hardware-specific kernels such as FlashAttention change the
    memory-traffic profile of the op itself.

    **System level.** Cache computed results and keys and values, keep the model warm
    so no request pays for loading, and for LLMs use **speculative decoding**, where a
    small draft model proposes tokens that the large model verifies in parallel,
    cutting latency without changing the output distribution.

    !!! interview "Staff move"
        Profile before you optimise, and say which axis you are spending. "First I
        find the actual bottleneck: compute, memory bandwidth, or data movement and
        preprocessing. Teams routinely optimise FLOPs while they are
        memory-bandwidth-bound, which is why quantization, which cuts bandwidth, often
        beats pruning, which cuts FLOPs. On device, int8 plus a hardware-targeted
        compile is the highest-leverage move, and I measure the accuracy-latency
        Pareto curve instead of chasing the fastest config. The trade-off to state out
        loud: aggressive quantization spends quality, so I hold a quality bar and find
        the fastest configuration under it, rather than finding the fastest
        configuration and discovering the quality cost in production."

    **Goes deeper:** [inference systems](../part14-systems/03-inference-systems.md)
    and [quantization](../part06-llm-training/05-quantization.md).

### 2. How to increase throughput {#q2}

??? question "Q: How do you increase throughput, and how does it trade against latency?"
    **The answer.** Throughput is requests or tokens per second across the fleet, a
    different objective from latency, and some of its techniques actively hurt latency.

    * **Batching** is the main lever. GPUs are throughput machines, and batching
      amortises kernel launch and weight loading across many requests. Dynamic or
      continuous batching groups requests arriving within a window. It **adds**
      latency, because a request waits for the batch, which is the trade stated
      plainly.
    * **Replication and sharding.** Data-parallel replicas scale horizontally; model
      parallelism is for models that do not fit one device.
    * **Asynchronous, pipelined serving** with an explicit queue, so a slow stage does
      not stall the fast ones.
    * **Smaller or quantized models** raise throughput too, since more requests fit in
      memory and each costs less.
    * **For LLMs, continuous batching plus a paged KV cache**, which packs cache
      memory into fixed blocks and removes the fragmentation that limits batch size
      ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)).

    !!! interview "Staff move"
        State the objective as a constrained optimization and name the two regimes.
        "Throughput and latency trade through batch size, so I never optimise one
        without a constraint on the other. The real objective is almost always
        maximise throughput subject to p99 latency under some budget, and the knobs
        are maximum batch size and batching window. I would also separate the regimes:
        a real-time perception path on-device cares about per-frame latency and cannot
        batch across requests at all, while an offline scoring or cloud path should
        batch aggressively. Same model, opposite serving strategy, and teams get into
        trouble by applying one team's playbook to the other's problem."

    **Goes deeper:** [inference systems](../part14-systems/03-inference-systems.md).

### 3. Caching trade-offs {#q3}

??? question "Q: What are the trade-offs in caching for ML serving?"
    **The answer.** Caching trades memory and staleness for latency and compute. Three
    forms with different profiles.

    * **Result caching.** Store outputs for repeated identical inputs. Excellent when
      the input distribution is skewed, as with popular queries, and useless when
      inputs are near-unique. The risk is staleness when the model or the correct
      answer changes.
    * **Feature caching.** Precompute expensive features such as embeddings offline
      and serve them from a store. Trades freshness for latency, and creates a
      training-serving skew risk when the cached features drift from how they are
      computed at training time.
    * **The KV cache.** Cache past keys and values so each new token attends over the
      cached prefix instead of recomputing it, which turns generation from $O(n^3)$
      into $O(n^2)$ over a length-$n$ sequence. The cost is memory that grows with
      sequence length times batch, which becomes the binding constraint on batch size
      and therefore on throughput. Paged attention, grouped-query attention and KV
      quantization all exist to push that constraint back.

    !!! interview "Staff move"
        Give the decision rule, then the operational failure nobody budgets for. "The
        decision is driven by two quantities: the entropy of the input distribution
        and the tolerance for staleness. High request skew plus tolerance for a
        slightly stale answer means cache aggressively; near-unique inputs or strict
        freshness means a cache just burns memory. The KV cache is a different animal,
        since it is the difference between linear and quadratic generation cost, so the
        engineering problem is managing its footprint, which is what paged attention
        is for. The part teams underinvest in is invalidation: a stale cache after a
        model update silently serves the old model's predictions, and no dashboard
        shows it unless you version the cache key by model."

    **Goes deeper:** [efficient attention and the KV cache](../part06-llm-training/04-efficient-attention-kv-cache.md)
    and [inference systems](../part14-systems/03-inference-systems.md).

### 4. What a staff engineer should know about large-scale training {#q4}

??? question "Q: What should a staff engineer know about large-scale training?"
    **The answer.** The parallelism taxonomy and where each applies.

    * **Data parallelism.** Replicate the model, split the batch, all-reduce the
      gradients. Scales until the global batch gets large enough to hurt optimization,
      and the communication cost is the gradient all-reduce.
    * **Tensor (model) parallelism.** Split individual layers across devices when the
      model does not fit. High communication, so it stays within a node with a fast
      interconnect.
    * **Pipeline parallelism.** Split layers into stages across devices and use
      micro-batches to keep the stages busy, paying a bubble at the start and end of
      each step.
    * **ZeRO and FSDP.** Shard optimizer states, gradients and parameters across the
      data-parallel workers, which fits large models without full model parallelism.
      The modern default.
    * **Mixed precision**, with bf16 preferred over fp16 for its fp32 exponent range.
    * **Gradient checkpointing**, trading recompute for activation memory.
    * **Gradient accumulation**, simulating a large batch under a memory limit.

    !!! interview "Staff move"
        Give the order you reach for them and then the non-obvious cost. "Mixed
        precision and gradient checkpointing first, because they are cheap and buy a
        lot of memory. Then FSDP or ZeRO for sharding. Then pipeline or tensor
        parallelism only when model size forces it, because that is where the
        engineering cost and the debugging difficulty jump. The staff-level point is
        that scaling the batch is not free: large-batch training changes the
        optimization, needs learning-rate scaling and warmup, and has a documented
        generalisation cost, so 'just add GPUs' has a statistical price and not only
        an infrastructure one. I also instrument model FLOPs utilization from day one,
        because without it you cannot tell whether you are compute-bound or
        communication-bound and you will optimise the wrong half."

    **Goes deeper:** [distributed training](../part14-systems/01-distributed-training.md)
    and [training systems](../part14-systems/02-training-systems.md).

---

## Data engines and real-world data

### 5. Auto-labeling and building a data engine {#q5}

??? question "Q: How do you approach auto-labeling and building a data engine?"
    **The answer.** The reframe first: at scale the bottleneck is labelled data, and
    human labelling does not scale linearly with data collection. A data engine is the
    closed loop that turns that into a flywheel, which is the shape of Tesla's data
    engine, of AV perception pipelines generally, and of SAM's billion-mask pipeline.

    The loop:

    1. **Seed model.** Train on a small human-labelled set.
    2. **Auto-label and mine.** Run the model over the unlabelled pool to produce
       pseudo-labels, and mine for the cases that matter: failures, disagreements,
       rare classes, edge cases.
    3. **Triage by uncertainty.** Do not label randomly. Use model signals to surface
       the highest-value examples, which is active learning (question 6).
    4. **Human in the loop.** Humans label or correct the mined hard cases and audit a
       sample of the auto-labels, so human attention concentrates where the model is
       weak.
    5. **Retrain** on the enlarged dataset.
    6. **Validate, ship, monitor**, and feed production failures back into step 2.

    **Auto-labeling techniques.** Pseudo-labelling with a confidence threshold;
    **ensemble agreement**, labelling where models agree and routing disagreement to
    humans; **temporal and multi-view consistency**, propagating labels across frames
    or cameras, since a detection confirmed from several viewpoints is trustworthy,
    which is heavily used in AV perception; **foundation models as labellers**, SAM for
    masks and a VLM for tags; and the **offline oracle**, a heavy, slow, high-accuracy
    model that auto-labels training data for the fast online model.

    !!! interview "Staff move"
        Lead with the asymmetric-oracle trick, which is the part that sounds like
        experience. "The frame is: spend human attention where the model is uncertain
        and let the model label where it is confident, which is active learning plus
        pseudo-labelling inside a monitored loop. The move I would emphasise is the
        asymmetric oracle: use an offline model that has advantages the online model
        cannot have, future frames, multiple views, no latency budget, much larger
        capacity, to generate training labels for the cheap online model. That is how
        AV companies bootstrap perception, and it converts a latency constraint into a
        labelling advantage. And the loop is the product: a static dataset rots under
        shift, so the deliverable is mine, triage, label, retrain, monitor, not any
        single checkpoint."

    **Goes deeper:** [weak supervision and auto-labeling](../part10-self-supervised/03-weak-supervision-and-auto-labeling.md),
    [Tesla](../part18-company-deep-dives/tesla.md) and
    [Scale AI and data engines](../part18-company-deep-dives/scale-ai-data-engines.md).

### 6. Choosing which examples to label {#q6}

??? question "Q: How do you choose which examples to label?"
    **The answer.** The budget is finite, so the objective is model improvement per
    label. The signals, and what each one catches:

    * **Uncertainty.** Softmax margin between the top two classes, or predictive
      entropy. Catches examples near the decision boundary. Cheap, and it
      over-selects redundant near-duplicates.
    * **Ensemble disagreement (query by committee).** Train several models, or use
      MC-dropout or different seeds, and select where they disagree most. More robust
      than single-model confidence because disagreement captures **epistemic**
      uncertainty, which more labels can fix.
    * **Mutual information (BALD).** Separates epistemic from **aleatoric**
      uncertainty explicitly, so you select what is learnable instead of what is
      inherently ambiguous.
    * **Diversity and coverage.** Pure uncertainty sampling clusters on one hard region
      and labels near-duplicates of each other. Add a coverage term, such as core-set
      selection or clustering in embedding space, so the labelled batch spans the
      input space. The strong methods balance uncertainty against diversity.
    * **Out-of-distribution distance.** Select examples far from the training
      distribution in embedding space, which is how drift monitoring feeds the
      labelling queue.
    * **Class-targeted selection.** Select for classes with poor per-class recall or
      for rare classes you are starved on, so active learning does not quietly ignore
      the tail.
    * **Model-label disagreement.** High disagreement between the model and the
      existing label flags a likely mislabel, which routes to re-annotation rather
      than to new labelling.

    !!! interview "Staff move"
        Name the failure mode of the naive version, then the policy you would actually
        run. "Naive uncertainty sampling collapses onto a narrow ambiguous region and
        labels near-duplicates, so I always pair an uncertainty signal with a coverage
        signal. I also separate epistemic from aleatoric uncertainty, because I want
        examples the model can learn from, and inherently ambiguous examples just
        inject label noise at high cost. In practice: ensemble disagreement plus
        embedding diversity, with explicit class balancing so the rare classes I care
        about do not get starved, and a confident-learning pass over what I already
        have to catch mislabels. The selection policy is a design artifact with its
        own trade-offs and its own evaluation, not a one-line heuristic."

    **Goes deeper:** [semi-supervised learning](../part10-self-supervised/02-semi-supervised.md)
    and [uncertainty and reliability](../part13-retrieval-eval-reliability/03-uncertainty-reliability.md).

### 7. The long tail, edge cases and shift {#q7}

??? question "Q: How do you deal with the realities of real-world data: the long tail, edge cases, and shift?"
    **The answer.** Five truths and the response each demands.

    * **The long tail dominates the remaining difficulty.** Getting to 90% is easy and
      the last 10% is brutal, because it is a long tail of rare, diverse cases: the
      unusual sign, the occluded pedestrian, the strange lighting. Average metrics hide
      it, so you need **slice-based evaluation** per class, per condition, per
      scenario. A model can have excellent aggregate performance and fail
      catastrophically on one safety-critical slice.
    * **Edge cases are mined, not anticipated.** You cannot enumerate them in advance.
      The engine mines them from production failures and disagreements, and every fixed
      edge case becomes a permanent regression test.
    * **Shift is the steady state.** New sensors, locations, seasons, behaviours.
      Design for continuous retraining and drift monitoring, not for a one-time train.
    * **Labels are noisy and the taxonomy is fuzzy.** Annotation guidelines are a
      living artifact, and an ambiguous class is a taxonomy problem to fix upstream
      rather than noise to model around.
    * **Costs are asymmetric.** A missed pedestrian and a false positive on a plastic
      bag are not the same event. Evaluation and loss have to encode that asymmetry,
      and the operating point is chosen for the worst-case slice.

    !!! interview "Staff move"
        Name one practice and one proportion. "The practice I would name is
        slice-based evaluation wired to a regression suite, because aggregate metrics
        lie about the tail, and in a safety-relevant system the question is not what
        the mAP is but what the recall is on the rarest safety-critical slice and
        whether it is monitored. My model of real-world vision: getting to a demo is a
        modelling problem, getting to production is a data and evaluation problem, and
        staying in production is a monitoring and data-engine problem. The model is a
        minority of the work, and the failure-mining loop and the slice evaluation are
        the majority, which is what separates a research result from a deployed
        system."

    **Goes deeper:** [AV perception system design](../part17-ml-system-design/05-perception-system-av.md)
    and [uncertainty and reliability](../part13-retrieval-eval-reliability/03-uncertainty-reliability.md).

### 8. Sensor fusion and multimodal perception {#q8}

??? question "Q: How does sensor fusion work, and what are the design choices?"
    **The answer.** Combine modalities with complementary strengths: cameras give rich
    semantics and no direct depth and fail in low light; radar gives range and radial
    velocity through weather and is sparse; lidar gives precise 3D geometry at higher
    cost and degrades in rain and spray. The design choice is **where** you combine.

    * **Early fusion** at the raw or low-feature level. Maximum access to cross-modal
      correlation, and maximum sensitivity to calibration error, time
      synchronisation, and any sensor dropping out.
    * **Late fusion** at the decision level. Each modality produces its own prediction
      and you combine at the end. Robust to a sensor failing and modular to develop,
      and it leaves cross-modal interaction on the table.
    * **Mid-level or deep fusion.** Combine intermediate features, through
      cross-attention between modalities or by projecting everything into a shared
      bird's-eye-view frame. The modern default, because it keeps most of the
      expressivity with better failure behaviour.

    The recurring engineering problems: **spatial and temporal calibration**, since the
    sensors must agree about where and when; **graceful degradation** when a modality
    drops out or degrades; and **asymmetric reliability**, trusting radar more in fog
    and the camera more in clear daylight, ideally as a learned content-dependent
    weighting instead of a fixed rule.

    !!! interview "Staff move"
        Name the axis, then the thing that actually breaks in the field. "The
        fusion-level choice is expressivity against robustness: early fusion captures
        the most cross-modal signal and is fragile to calibration and dropout, late
        fusion is robust and modular and leaves signal unused, and mid-level fusion in
        a shared spatial frame is where the field landed. What I would stress from
        deployed systems is that synchronisation and calibration are where fusion
        actually fails, well before the model architecture matters, and that graceful
        degradation under sensor dropout has to be a first-class design requirement
        with its own tests. The system has to know which sensor to trust in which
        condition and keep operating when one is gone, and that requirement changes
        the architecture rather than being bolted on."

    **Goes deeper:** [sensor fusion](../part11-perception-autonomy/03-sensor-fusion.md)
    and [AV perception system design](../part17-ml-system-design/05-perception-system-av.md).

## References

* Kwon et al. "Efficient Memory Management for Large Language Model Serving with
  PagedAttention", SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
* Leviathan, Kalman, Matias. "Fast Inference from Transformers via Speculative
  Decoding", ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
* Kirillov et al. "Segment Anything", ICCV 2023, for the model-in-the-loop data engine
  that produced SA-1B. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
* Northcutt, Jiang, Chuang. "Confident Learning: Estimating Uncertainty in Dataset
  Labels", JAIR 2021. [arXiv:1911.00068](https://arxiv.org/abs/1911.00068)
