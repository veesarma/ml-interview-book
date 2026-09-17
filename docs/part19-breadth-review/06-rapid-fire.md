# Rapid-fire derivations and delivery

> **Why this matters at staff level.** These are the things you produce on a
> whiteboard without hesitation. Not because an interviewer will ask for all of them,
> but because when one arrives mid-answer you want to derive it in fifteen seconds and
> keep talking, instead of stalling and losing the thread of the larger question you
> were actually being asked. Drill them until they are reflexive, then spend the rest
> of your preparation on delivery, which is the other half of the score.

## How to use this chapter

Each card is one derivation or one-liner. Cover the page, read the prompt in the
heading, produce the answer out loud or on paper, then open the card. Anything you
cannot produce in fifteen seconds goes on tomorrow's list. A full pass takes about
25 minutes once warm.

The three sections after the cards are about delivery: the answer skeleton, the three
habits, and a short list of the sentences that cost candidates offers.

---

## Losses and probability

??? question "Card: gradient of sigmoid cross-entropy with respect to the logit"
    For $p = \sigma(z)$ and binary target $y$, with
    $L = -[y \log p + (1-y)\log(1-p)]$:

    $$
    \frac{\partial L}{\partial p} = \frac{p - y}{p(1-p)}, \qquad \frac{\partial p}{\partial z} = p(1-p)
    \;\Longrightarrow\; \boxed{\;\frac{\partial L}{\partial z} = p - y\;}
    $$

    The $p(1-p)$ cancels, which is why cross-entropy keeps a large gradient when the
    model is confidently wrong and MSE on a sigmoid stalls there.

??? question "Card: gradient of softmax cross-entropy"
    For logits $z \in \R^K$, $p = \softmax(z)$, one-hot $y$:

    $$
    \boxed{\;\frac{\partial L}{\partial z_i} = p_i - y_i\;}
    $$

    Same form as the binary case. The derivation uses
    $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$, and the sum over $j$
    collapses because $\sum_j y_j = 1$.

??? question "Card: numerically stable softmax"
    $\softmax(z) = \softmax(z - \max_j z_j)$, because shifting all logits by a
    constant multiplies numerator and denominator by the same factor. Subtract the max
    before exponentiating and the largest exponent becomes $e^0 = 1$, so nothing
    overflows. The same trick is log-sum-exp:
    $\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}$ with $m = \max_j z_j$.

??? question "Card: cross-entropy, entropy and KL"
    $H(p, q) = H(p) + \KL(p \,\|\, q)$. Training with cross-entropy against a fixed
    empirical distribution minimises the KL from your model to the data, since $H(p)$
    does not depend on the parameters. This is also why cross-entropy loss on one-hot
    labels equals the negative log-likelihood of the correct class.

??? question "Card: L1 and L2 as priors"
    A regularised loss is a negative log posterior. **L2 is a Gaussian prior**,
    $-\log \mathcal{N}(0, \tau^2) \propto \norm{w}_2^2$. **L1 is a Laplace prior**,
    $-\log \mathrm{Laplace}(0, b) \propto \norm{w}_1$. The L1 constraint region is a
    cross-polytope whose corners lie on the axes, so the first contact between the loss
    contours and the region is usually at a corner, where some coordinates are exactly
    zero.

??? question "Card: focal loss"
    $\mathrm{FL}(p_t) = -(1-p_t)^\gamma \log p_t$, with $p_t$ the probability assigned
    to the true class. The factor $(1-p_t)^\gamma$ goes to zero for easy examples
    (high $p_t$), so the gradient concentrates on hard and rare ones. It was built for
    the foreground-background imbalance of dense detection, where one image contains on
    the order of 100k easy background anchors. An $\alpha$ term is usually added for
    per-class weighting.

??? question "Card: InfoNCE"
    For a positive pair $(x, x^+)$ and negatives $x^-_j$ with similarity $s$ and
    temperature $\tau$:

    $$
    L = -\log \frac{\exp(s(x, x^+)/\tau)}{\exp(s(x, x^+)/\tau) + \sum_j \exp(s(x, x^-_j)/\tau)}
    $$

    A cross-entropy over similarities: pull the positive together, push the negatives
    apart. In CLIP it is applied symmetrically over the image-to-text and text-to-image
    directions of the batch similarity matrix. The augmentations (or the pairing rule)
    define which invariances the representation learns.

??? question "Card: hinge loss"
    $L = \max(0, 1 - y f(x))$ with $y \in \{-1, +1\}$. Zero gradient once a point is
    correctly classified with margin at least 1, which is what makes the solution
    depend only on the support vectors, and which is also why it gives you a score
    rather than a calibrated probability.

## Statistics, evaluation, ensembles

??? question "Card: bias-variance decomposition"
    $$
    \E\big[(y - \hat f(x))^2\big] = \mathrm{Bias}[\hat f(x)]^2 + \mathrm{Var}[\hat f(x)] + \sigma^2
    $$

    Underfit is high bias, overfit is high variance, and $\sigma^2$ is the noise in
    $Y \mid X$ that no model removes. The decomposition is exact for squared loss; for
    0-1 loss the analogues are approximate, which is worth saying if pushed.

??? question "Card: variance of an average of correlated estimators"
    $B$ estimators, each with variance $\sigma^2$, pairwise correlation $\rho$:

    $$
    \boxed{\;\mathrm{Var}\Big(\tfrac{1}{B}\textstyle\sum_b \hat f_b\Big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2\;}
    $$

    The floor as $B \to \infty$ is $\rho \sigma^2$, so decorrelation is the lever and
    tree count only has to be past diminishing returns. Feature subsampling attacks
    $\rho$ directly, which is why random forests beat bagged trees.

??? question "Card: bootstrap out-of-bag fraction"
    The probability a given row is missed by a bootstrap sample of size $n$ is
    $(1 - 1/n)^n \to e^{-1} \approx 0.368$. So each tree never saw about 37% of the
    data, which gives you a validation estimate without holding anything out.

??? question "Card: AUC as a rank statistic"
    $\mathrm{AUC} = P(s^+ > s^-) + \tfrac12 P(s^+ = s^-)$: the probability a random
    positive outranks a random negative, with ties counted as half. It is invariant to
    prevalence and to the threshold, which is the property that makes it useful for
    model comparison and useless as a proxy for a business metric measured at one
    operating point. Baseline is 0.5 whatever the class balance.

??? question "Card: PR-AUC baseline"
    A random classifier's precision-recall curve is a horizontal line at the positive
    rate $\pi$, so PR-AUC baseline is $\pi$ and not 0.5. Under 1:1000 imbalance, a
    PR-AUC of 0.10 is a hundredfold lift over chance, and an ROC-AUC of 0.95 on the
    same model can mean very little.

??? question "Card: the cost-optimal threshold"
    For a calibrated probability $p$ with false-positive cost $c_{\text{FP}}$ and
    false-negative cost $c_{\text{FN}}$, predict positive when

    $$
    p \ge \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}}
    $$

    The default 0.5 is the special case of equal costs, which is almost never your
    situation. This is the sentence that turns a metric discussion into a business
    discussion.

??? question "Card: the three kinds of shift"
    **Covariate shift**: $P(X)$ moves, $P(Y \mid X)$ stable. **Label or prior shift**:
    $P(Y)$ moves. **Concept drift**: $P(Y \mid X)$ moves. The remedy differs per kind:
    importance reweighting, prior correction in closed form, and retraining
    respectively, which is why naming the kind is the first move.

## Networks and training

??? question "Card: He and Xavier initialization"
    Hold the activation variance constant through depth. For fan-in $n_{\text{in}}$:
    **Xavier** uses $\mathrm{Var}(W) = 1/n_{\text{avg}}$ over fan-in and fan-out, for
    tanh-like regimes. **He** uses $\mathrm{Var}(W) = 2/n_{\text{in}}$ for ReLU, where
    the 2 compensates for ReLU zeroing half of its inputs and so halving the variance.
    Biases at zero, because the random weights already broke the symmetry.

??? question "Card: why gradients vanish or explode"
    Backprop multiplies local Jacobians along the path. If the product of their
    singular values is below 1 the gradient vanishes, above 1 it explodes. The three
    standard fixes attack it from three directions: init controls the initial scale,
    normalization keeps it controlled during training, and a residual connection adds
    an identity path whose contribution does not attenuate.

??? question "Card: BatchNorm vs LayerNorm vs RMSNorm in one line each"
    **BN** normalizes per channel across the batch; differs between training (batch
    statistics) and inference (running statistics); breaks under small, correlated or
    non-stationary batches. **LN** normalizes per example across features; identical
    in both modes; the Transformer standard. **RMSNorm** is LN without mean-centring,
    dividing by the root mean square only; cheaper, and the current LLM default.

??? question "Card: inverted dropout"
    At training time, zero each activation with probability $1-q$ and divide the
    survivors by the keep probability $q$, so the expected activation is unchanged.
    Inference is then an ordinary forward pass with nothing to scale. Interviewers who
    learned it from the 2014 paper may expect test-time scaling; the frameworks all do
    it at training time.

??? question "Card: what AdamW changes"
    In Adam, an L2 penalty added to the loss is scaled by the adaptive per-parameter
    denominator, so parameters with large gradient magnitude get less decay than
    intended. AdamW applies $\theta \leftarrow \theta - \eta\lambda\theta$ as a
    separate step, decoupled from the adaptive update, which restores uniform decay and
    measurably improves generalisation.

??? question "Card: the linear learning-rate scaling rule"
    When you multiply the batch size by $k$, multiply the learning rate by $k$ (or by
    $\sqrt{k}$ under the square-root variant) and add warmup over the first few hundred
    to few thousand steps. Warmup exists because the early adaptive moment estimates
    are noisy and the early gradients are large, which is exactly when a scaled-up
    learning rate diverges.

??? question "Card: gradient checkpointing"
    Store activations only at checkpoint boundaries and recompute the rest during the
    backward pass: memory drops from $O(L)$ toward $O(\sqrt{L})$ for $L$ layers with
    evenly spaced checkpoints, at the cost of roughly one extra forward pass. It is the
    standard memory-for-compute trade, and it is the first thing to turn on when a
    training run will not fit.

## Attention and Transformers

??? question "Card: attention, with shapes"
    $Q \in \R^{T \times d_k}$, $K \in \R^{T \times d_k}$, $V \in \R^{T \times d_v}$:

    $$
    \boxed{\;\mathrm{Attention}(Q,K,V) = \softmax\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V\;}
    $$

    The score matrix is $T \times T$, so attention is $O(T^2 d)$ in time and, without
    FlashAttention, $O(T^2)$ in memory. The position-wise FFN is $O(T d^2)$ and usually
    dominates the FLOP count when $d > T$.

??? question "Card: why $1/\sqrt{d_k}$"
    Components independent, zero mean, unit variance: each product term has variance 1,
    and the dot product of $d_k$ of them has variance $d_k$, so a standard deviation of
    $\sqrt{d_k}$. Large logits saturate the softmax, the Jacobian goes to zero, and
    learning stops. Dividing holds the logit variance at 1. It is also a fixed
    temperature, which is the bridge to distillation and decoding.

??? question "Card: the KV cache"
    Cache past keys and values so generating token $t$ attends over $t$ cached
    positions at $O(td)$ instead of recomputing the prefix. Total for a length-$n$
    generation: $O(n^2 d)$ with the cache against $O(n^3 d)$ without. Cache size in
    bytes:

    $$
    2 \times L \times n_{\text{kv heads}} \times d_{\text{head}} \times T \times B \times \text{bytes per element}
    $$

    The leading 2 is keys plus values. That expression is why GQA, MLA, paged
    attention and cache quantization all exist.

??? question "Card: RoPE in one line"
    Rotate $q$ and $k$ by an angle proportional to their absolute position. A rotation
    preserves inner products up to the relative angle, so the resulting score depends
    only on $m - n$. Relative-aware, parameter-free, and defined at any position, which
    is why it degrades more gracefully past the training length than a learned table,
    which has no entry at all.

??? question "Card: pre-norm against post-norm"
    Pre-norm is $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LN}(x))$, leaving the
    residual path a clean identity, which is what makes very deep stacks stable.
    Post-norm is $x \leftarrow \mathrm{LN}(x + \mathrm{Sublayer}(x))$, which puts a
    rescaling on the main path, can reach marginally better quality, and is fragile at
    depth. Every large modern model is pre-norm, because divergence on a multi-week run
    is unaffordable.

??? question "Card: SwiGLU sizing"
    $\mathrm{SwiGLU}(x) = \big(\mathrm{Swish}(xW_1)\big) \odot (xW_3)$, then project
    with $W_2$: three matrices instead of two. To stay compute-neutral against a
    $4d$-wide two-matrix FFN, the hidden width is set to about $\tfrac{8}{3}d$. The
    gain is quality per FLOP, and the failure mode is gate collapse, the multiplicative
    analogue of a dead ReLU.

??? question "Card: multi-head attention, redundancy, sinks"
    $h$ heads of width $d/h$ attend in parallel subspaces and concatenate, so several
    kinds of relation are captured at close to single-head cost. Empirically many heads
    are prunable with little loss, and some heads park probability mass on the first
    token, the attention sink, which breaks window attention in streaming inference and
    distorts activation ranges under quantization.
    Head redundancy is also the reason sharing keys and values across heads works.

??? question "Card: what actually breaks at 4x the training length"
    Three separate failures, and candidates usually name only the first. Positions or
    rotation phases never seen in training, so the scores are out of distribution.
    Attention entropy dilution, since softmax over four times as many keys spreads the
    mass thinner than the model was trained to handle. And the content dependencies at
    those distances were never in the training data at all. Interpolation, YaRN and
    ALiBi address the first, not the third.

## The LLM stack

??? question "Card: Chinchilla, with the inference correction"
    Compute-optimal training scales parameters and tokens in equal proportion, around
    20 tokens per parameter for the budgets studied, and many earlier large models were
    parameter-heavy and data-starved. The correction to carry: that ratio optimises
    **training** compute only. When you serve a model to many users, inference
    dominates total cost, so production deliberately overtrains smaller models well
    past 20:1 to get a permanently cheaper thing to serve.

??? question "Card: MHA, MQA, GQA, MLA"
    One pressure, four answers. MHA caches $h$ key-value pairs. MQA shares one pair
    across all heads, cutting the cache by $h$ and costing quality. GQA groups the
    query heads into $g$ groups sharing a pair, with $g \approx 8$ recovering most of
    the quality; the modern default. MLA caches a low-rank latent and reconstructs
    per-head keys and values, compressing further while preserving more per-head
    expressivity.

??? question "Card: LoRA"
    $W' = W + BA$ with $B \in \R^{d \times r}$, $A \in \R^{r \times k}$, $r \ll d, k$.
    $A$ random, $B$ zero, so the update starts at exactly zero and training begins from
    the pretrained function. Merge $BA$ into $W$ at inference for zero added latency,
    or keep many small adapters against one frozen base. QLoRA adds a 4-bit frozen
    base.

??? question "Card: MoE arithmetic"
    Total parameters set the memory footprint, active parameters set the per-token
    FLOPs, and the router picks the top-$k$ of $E$ experts per token. The auxiliary
    load-balancing loss is what stops the router collapsing onto a few experts and
    wasting the rest. A compute saver paid for in memory and in serving complexity.

??? question "Card: why decode is memory-bandwidth-bound"
    Generating one token reads every weight from memory and does one token's worth of
    arithmetic, so the arithmetic intensity is about 1 FLOP per byte, far below what
    the hardware needs to be compute-bound. Prefill, which processes the whole prompt
    at once, is compute-bound. That single asymmetry explains why quantization, GQA,
    paged KV caches and FlashAttention all target bytes moved, and why batching raises
    throughput without helping single-request latency.

??? question "Card: speculative decoding"
    A small draft model proposes $k$ tokens, the large model verifies them in one
    parallel forward pass, and a rejection-sampling correction keeps the output
    distribution identical to sampling from the large model alone. It converts
    bandwidth-bound sequential steps into one batched step, so the speedup tracks the
    draft model's acceptance rate.

??? question "Card: Little's law for serving"
    Concurrency $=$ throughput $\times$ latency. A service holding 64 in-flight
    requests at 200 ms each sustains 320 requests per second, and if you want more
    throughput at fixed latency you need more concurrency, which means more memory for
    the KV cache. It converts a serving argument into arithmetic you can do on a
    whiteboard.

## Vision and multimodal

??? question "Card: the residual connection"
    $y = x + F(x)$. The identity path gives the gradient a route that is not attenuated
    by the block's Jacobian, which is what made 100-layer networks trainable and what
    pre-norm Transformers reuse. Same idea, two eras, and saying that out loud connects
    your vision background to the LLM stack.

??? question "Card: IoU and NMS"
    IoU is intersection over union of two boxes. NMS sorts detections by score and
    suppresses any box whose IoU with a kept box exceeds a threshold. It is
    non-differentiable, threshold-sensitive, and it fails in crowded scenes where two
    true objects genuinely overlap, which is the motivation for DETR's set prediction
    with Hungarian matching.

??? question "Card: ViT in three sentences"
    Split the image into fixed patches, embed each linearly as a token, add positional
    embeddings, run a Transformer encoder. Dropping the locality and translation
    priors costs accuracy on small datasets and pays off with enough data or
    pretraining. Swin puts locality and hierarchy back through shifted windows;
    ConvNeXt showed a modernised CNN matches ViT under the same training recipe.

??? question "Card: CLIP zero-shot classification"
    Embed the image with the image tower. Embed each candidate class as a text prompt
    ("a photo of a {class}") with the text tower. Take the nearest by cosine
    similarity. No fine-tuning, an arbitrary label set at inference time, and it is the
    mechanism behind open-vocabulary detection and segmentation.

??? question "Card: MAE"
    Mask about 75% of image patches, encode only the visible ones, and reconstruct the
    missing pixels with a lightweight decoder. The high mask ratio is what makes the
    task non-trivial, and encoding only the visible patches is what makes it fast. The
    BERT idea for vision, and strong ViT pretraining.

---

## The reusable answer skeleton

For any open-ended question, say the headline first, then walk these five steps out
loud. The ordering is the thing being graded, and it is the same skeleton from the
[part index](index.md).

| Step | What you say | Why it scores |
|---|---|---|
| 1. Define | "By calibration I mean that among predictions of 0.7, 70% are positive." | Catches the case where you and the interviewer mean different things, and buys three seconds |
| 2. Standard answer | The mechanism, with the equation if there is one | This is the L5 bar and you clear it in 30 seconds |
| 3. Trade-off | "The axis I am spending here is quality for latency" | Shows you think in trade-offs and not in right answers |
| 4. Failure mode | "What bites you is that a temperature fit in July does not hold in November" | The highest-signal move available, and the hardest to fake |
| 5. Commit | "Given a 30 ms budget I take the int8 model and hold the quality bar at X" | Seniority is the willingness to decide under a stated context |

A worked contrast on one question, "should we use RAG or fine-tune?":

* **Weak:** "It depends on the use case. RAG is good for knowledge and fine-tuning is
  good for style. Both have pros and cons."
* **Strong:** "RAG, if the failure is a knowledge gap. Let me define the split: RAG
  injects knowledge at inference, fine-tuning changes behaviour. Your symptom is that
  it gets policy details wrong, which is knowledge, and your policies change monthly,
  which settles it, since re-indexing is cheap and retraining monthly is not. The
  trade is latency, since you pay retrieval plus a longer context, and the failure
  mode is that RAG's ceiling is the retriever, so I would evaluate retrieval
  separately with recall@k before touching the prompt. If the model also writes in the
  wrong register, that part is fine-tuning, and I would do both."

Same content. One of them gets an offer.

## Three habits to rehearse

**State the headline, then earn it.** "Cross-entropy, because the sigmoid derivative
cancels and learning does not stall when the model is confidently wrong. Let me show
you." Do not make the interviewer sit through a derivation to find out what you
concluded. In a 45-minute round with twelve questions, the interviewer is taking notes
on your first sentence.

**Make the trade-off explicit, then commit.** The weak answer stops at "it depends".
The staff answer is "it depends on X, here is the dependency, and given a
latency-critical on-device context I would choose Y and accept Z". Name the axis you
are spending: quality, latency, memory, compute, robustness, maintainability. Then
pick a side once the context is stated, and say what would change your mind.

**Volunteer the failure mode.** The single highest-signal move available is surfacing
the sharp edge before you are asked. "And the thing that will bite you here is
training-serving skew, so I would log-and-replay before I trusted the offline number."
That sentence is the difference between someone who has read about a technique and
someone who has been paged at 3am because of it, and interviewers can tell which one
they are hearing.

## Sentences that cost people offers

* "That's just a hyperparameter." (Said about something that decides whether the run
  converges.)
* "It's basically the same as X." (When asked to compare two things, the differences
  are the question.)
* "I'd A/B test it." (As a substitute for having an opinion, rather than as the last
  step after having one.)
* "We used the standard approach." (Which approach, chosen over what, and why.)
* "I'd need to look that up." (Fine for a citation or an exact number. Not fine for a
  mechanism you claim to have shipped.)
* A five-minute answer to a two-minute question, which costs you the three follow-ups
  where the real signal was going to be.

You know this material. The round is about demonstrating that you reason from
principles, think in trade-offs, and know where the bodies are buried.
