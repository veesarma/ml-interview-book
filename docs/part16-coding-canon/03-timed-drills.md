# Timed coding drills

> **Why this matters at staff level.** A coding round is 45 minutes with one pass,
> no debugger worth the name, and someone asking questions while you type. The
> skill being measured is not whether you know attention. It is whether you can
> produce a correct, shaped, explained implementation under those conditions. That
> is a separate skill from knowing the algorithm and it responds to practice in a
> way that reading does not.

## TL;DR: the interview card

* Twenty-two drills at three tiers: warm-up (10 min), core (25 min), extended (45 min).
* Run them against a clock you can see. Stop when it rings even if you are three lines from done, then note what you were three lines from.
* Narrate. Silence reads as "stuck" whether or not you are, and the explanation is a graded axis.
* Write the shape comment before the line, always. It is worth more than the line.
* Run your code at minute 8, not minute 40, on `B=2, T=3, d=4`.
* Five axes on the rubric: correctness, shapes, numerics, API fluency, explanation. Four out of five is a strong attempt.
* The three mock sessions at the end combine drills into 60 minutes, which is the real format.

## How to run a drill

Set a timer for the tier's target. Open a blank file. Read the prompt once, then
say your plan aloud for 60 seconds before typing anything, because that is what the
interview requires and practising the code without the narration trains half the
skill.

Hints are in collapsible blocks at three levels. Take one only after three minutes
of no progress, and record which level you needed: the level is the score.

When the timer rings, run the acceptance test, then read the reference module and
write one line in your miss log. Then delete your file.

### Frequency, from most asked to least

Ordering from what gets asked in loops at autonomy companies, consumer-ranking
companies and frontier labs. The first five account for most coding rounds.

| Rank | Topic | Drills | Canon |
|---|---|---|---|
| 1 | Attention with a mask | W3, C1 | #25, #26, #27 |
| 2 | IoU and NMS | W1, C2 | #18, #19 |
| 3 | Backprop through one layer by hand | C3, E3 | #11, #12 |
| 4 | K-means | C4 | #5 |
| 5 | Softmax, cross-entropy, numerical stability | W2 | #3 |
| 6 | Sampling: temperature, top-k, top-p | C5 | #32 |
| 7 | BPE tokenizer | C9 | #24 |
| 8 | KV cache | C6 | #33 |
| 9 | LoRA | W5 | #38 |
| 10 | DPO loss | W6 | #55 |
| 11 | LayerNorm | W4 | #14 |
| 12 | GAE and PPO | W7, C7 | #52, #53 |
| 13 | Average precision and mAP | C8 | #59 |
| 14 | Convolution from scratch | E2 | #16 |
| 15 | Anchors, matching, focal loss | E4 | #20, #21 |
| 16 | A whole GPT forward pass | E1 | #29, #30, #31 |
| 17 | Reward model with a pairwise loss | E5 | #54 |
| 18 | GRPO with group advantages | E6 | #56 |

## Tier 1: warm-up, 10 minutes each

Ten-minute drills are the ones where a strong candidate should be done in six and
spend four explaining. They are also what to do the morning of an interview.

### W1: IoU

**The prompt, as you will hear it.** "Write me a function that takes two sets of
boxes in xyxy format and returns the pairwise IoU matrix. Vectorised, no loops."

**Time.** 10 minutes. **Canon #18.**

**Acceptance test.** `pytest tests/test_detection_boxes.py`. Or by hand: IoU of a
box with itself is 1, IoU of two disjoint boxes is 0, and `[0,0,10,10]` against
`[5,5,15,15]` is `25 / 175`.

??? tip "Hint 1"
    You need the intersection rectangle before you need anything else. Its
    top-left corner is the elementwise max of the two top-left corners, and its
    bottom-right is the elementwise min of the two bottom-rights.

??? tip "Hint 2"
    The pairwise part comes from broadcasting: `a[:, None, :2]` against
    `b[None, :, :2]` gives `(N, M, 2)`.

??? tip "Hint 3"
    The width and height of the intersection must be clamped at zero before you
    multiply them. Two disjoint boxes give a negative width and a negative height,
    and their product is positive.

### W2: softmax and cross-entropy, stable

**The prompt.** "Implement softmax and then cross-entropy loss from logits, in
NumPy, without calling anything from scipy. Then tell me what happens when a logit
is 1000."

**Time.** 10 minutes. **Canon #3.**

**Acceptance test.** `pytest tests/test_nn_losses.py`. By hand: rows sum to 1,
`softmax(x + c) == softmax(x)` for any scalar `c`, and a logit of 1000 does not
produce `NaN`.

??? tip "Hint 1"
    Softmax is shift-invariant. Use that.

??? tip "Hint 2"
    `keepdims=True` on the max and on the sum, so both broadcast back over the
    class axis.

??? tip "Hint 3"
    For the loss, do not compute probabilities and then take their log. Compute
    `z - logsumexp(z)` directly, then index the target class with
    `logp[np.arange(N), y]`.

### W3: single-head attention

**The prompt.** "Write scaled dot-product attention. Q, K, V are `(T, d)`. Add an
optional causal mask."

**Time.** 10 minutes. **Canon #25, #27.**

**Acceptance test.** `pytest tests/test_transformer_attention.py`, or compare
against `torch.nn.functional.scaled_dot_product_attention` on random inputs.

??? tip "Hint 1"
    Three lines: scores, softmax, weighted sum. Write the shapes first:
    `(T, S)`, `(T, S)`, `(T, d)`.

??? tip "Hint 2"
    The scale is `1 / sqrt(d_k)` where `d_k` is the last dimension of K. Be ready
    to say why: the dot product of two vectors with unit-variance entries has
    variance `d_k`, so without the scale the softmax saturates as `d_k` grows.

??? tip "Hint 3"
    The causal mask is `torch.tril(torch.ones(T, S, dtype=torch.bool))`, applied
    with `masked_fill` using a finite large negative value, before the softmax and
    not after.

### W4: LayerNorm

**The prompt.** "Implement LayerNorm. Then write its backward by hand, no autograd."

**Time.** 10 minutes forward, 20 with the backward. **Canon #14.**

**Acceptance test.** `pytest tests/test_nn_normalization.py`, or compare the
forward against `torch.nn.functional.layer_norm` and the backward against
`torch.autograd.grad` on float64 inputs.

??? tip "Hint 1"
    Normalise over the last axis, per token, with `keepdims=True` on both the mean
    and the variance. The epsilon goes inside the square root.

??? tip "Hint 2"
    Use the biased variance (`ddof=0`, `unbiased=False`). Torch's LayerNorm does,
    and a test against it fails by `n/(n-1)` otherwise.

??? tip "Hint 3"
    For the backward, let $\hat x = (x - \mu)/\sigma$ and $g = \partial L/\partial \hat x$.
    Then
    $$\frac{\partial L}{\partial x} = \frac{1}{\sigma}\left(g - \bar g - \hat x \,\overline{g \hat x}\right)$$
    where the bars are means over the normalised axis. The two correction terms
    are the paths through $\mu$ and through $\sigma$.

### W5: LoRA

**The prompt.** "Wrap a frozen `nn.Linear` with a LoRA adapter of rank r. Show me
that it is the identity at initialisation and that merging is free at inference."

**Time.** 10 minutes. **Canon #38.**

**Acceptance test.** `pytest tests/test_finetune_lora.py`. By hand: the output at
step 0 equals the frozen layer's output exactly, and `W + (alpha/r) * B @ A`
reproduces the adapted forward.

??? tip "Hint 1"
    `A` is `(r, d_in)`, `B` is `(d_out, r)`. Torch's `Linear` stores weight as
    `(d_out, d_in)`, so the adapter multiplies in that order.

??? tip "Hint 2"
    `B` is initialised to zeros. That is what makes the adapter the identity at
    step 0 and it is the first thing to check if fine-tuning spikes on step one.

??? tip "Hint 3"
    Freeze by `weight.requires_grad_(False)`, and remember that the frozen forward
    still needs to run under grad so the adapter's gradient can flow through the
    sum.

### W6: the DPO loss

**The prompt.** "Given the policy and reference log-probabilities of a chosen and a
rejected response, write the DPO loss. Then tell me what you would watch during
training."

**Time.** 10 minutes. **Canon #55.**

**Acceptance test.** `pytest tests/test_posttrain_dpo.py`. By hand: a policy equal
to the reference gives exactly `log 2`, and the loss decreases as the margin grows.

??? tip "Hint 1"
    Four scalars per example. The implicit reward of a response is
    `beta * (log pi(y) - log pi_ref(y))`, so the margin is the difference of two
    differences.

??? tip "Hint 2"
    `loss = -F.logsigmoid(beta * margin).mean()`. Never `-torch.log(torch.sigmoid(...))`:
    it overflows to `inf` once the margin passes about 30, which a working run
    reaches within a few hundred steps.

??? tip "Hint 3"
    What to watch is the margin, not the loss. The loss also falls when the policy
    pushes the rejected log-probability down without raising the chosen one, which
    is the failure mode where both responses get worse.

### W7: GAE

**The prompt.** "Implement generalised advantage estimation. Rewards, values and
done flags for one trajectory."

**Time.** 10 minutes. **Canon #52.**

**Acceptance test.** `pytest tests/test_rl_gae.py`. By hand: with `lam = 1` it
equals the Monte Carlo return minus the value; with `lam = 0` it equals the
one-step TD error.

??? tip "Hint 1"
    Compute the TD residuals first:
    $\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$.

??? tip "Hint 2"
    Then accumulate backwards:
    $A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}$, with $A_T = 0$.

??? tip "Hint 3"
    The `(1 - done)` factor appears twice, once in the residual and once in the
    recursion, and both are needed: a terminal state has no bootstrap value and no
    future advantage. Getting only one of them gives advantages that leak across
    episode boundaries.

## Tier 2: core, 25 minutes each

These are the standard 45-minute round with 20 minutes of conversation around
them.

### C1: multi-head attention with masks

**The prompt.** "Implement multi-head self-attention with a causal mask and
support for a padding mask. Batched. Separate Q, K, V projections. Then tell me the
FLOP and memory cost."

**Time.** 25 minutes. **Canon #26, #27.**

**Acceptance test.** `pytest tests/test_transformer_multihead.py`, or compare
against `F.scaled_dot_product_attention` with `is_causal=True`.

??? tip "Hint 1"
    Five shapes, in order: `(B, T, d)`, `(B, T, H, dh)`, `(B, H, T, dh)`,
    `(B, H, T, S)`, back to `(B, T, d)`. Write all five as comments first.

??? tip "Hint 2"
    The causal mask is `(T, S)` and the padding mask is `(B, 1, 1, S)`. Combine
    with `&`, then convert to an additive bias of shape `(B, 1, T, S)` so it
    broadcasts over heads.

??? tip "Hint 3"
    The merge is `ctx.transpose(1, 2).reshape(B, T, d)` and it must be `reshape`,
    not `view`, because `transpose` left it non-contiguous. For the cost: scores
    and weights are `4 B H T S` bytes each in fp32, and the FLOPs are about
    `4 B T^2 d` for the two matmuls plus `4 B T d^2` for the four projections.

### C2: NMS

**The prompt.** "Implement non-maximum suppression. Boxes, scores, an IoU
threshold. Then make it work per class."

**Time.** 25 minutes. **Canon #19.**

**Acceptance test.** `pytest tests/test_detection_nms.py`, or compare against
`torchvision.ops.nms` on random boxes at several thresholds.

??? tip "Hint 1"
    Sort by score descending, take the top box, suppress everything with IoU above
    the threshold against it, repeat on what is left.

??? tip "Hint 2"
    Keep an index array rather than mutating the boxes. `order = scores.argsort()[::-1]`,
    then in the loop `order = order[1:][iou <= threshold]`.

??? tip "Hint 3"
    For per class, add a per-class offset larger than the coordinate range to all
    four coordinates, run one global NMS, and the classes cannot interact. That is
    `torchvision.ops.batched_nms` and it beats a Python loop over classes by the
    number of classes.

### C3: backward through a linear and a ReLU

**The prompt.** "Two-layer MLP, NumPy, no autograd. Forward and backward. Then
verify your gradient numerically."

**Time.** 25 minutes. **Canon #11.**

**Acceptance test.** `pytest tests/test_nn_mlp.py`, or a central-difference check
with `h = 1e-5` on float64 agreeing to about 1e-7 relative.

??? tip "Hint 1"
    Cache what you need on the forward pass: the input, the pre-activation, and
    the post-activation of each layer.

??? tip "Hint 2"
    For $Y = XW + b$ with $X$ of shape $(N, d_{in})$:
    $\partial L/\partial X = (\partial L/\partial Y) W^\top$,
    $\partial L/\partial W = X^\top (\partial L/\partial Y)$,
    $\partial L/\partial b = \sum_n \partial L/\partial Y$.
    Check each by shape: the gradient has the shape of the thing it differentiates.

??? tip "Hint 3"
    ReLU's backward is a mask on the *pre-activation*, not the output. And if your
    loss averages over the batch, the `1/N` has to appear once, at the top of the
    backward pass, not once per layer.

### C4: K-means

**The prompt.** "Implement K-means. Then tell me how you would initialise it and
what happens when a cluster goes empty."

**Time.** 25 minutes. **Canon #5.**

**Acceptance test.** `pytest tests/test_classical_knn_kmeans.py`. By hand: the
objective decreases monotonically, and on three well-separated Gaussians it
recovers the clusters.

??? tip "Hint 1"
    Two steps in a loop: assign each point to its nearest centre, then set each
    centre to the mean of its points. Stop when assignments stop changing.

??? tip "Hint 2"
    The assignment step is the pairwise distance drill:
    `(X**2).sum(1, keepdims=True) - 2 * X @ C.T + (C**2).sum(1)` of shape `(N, K)`,
    then `argmin(axis=1)`.

??? tip "Hint 3"
    An empty cluster gives `0/0` in the update. Reseed it at the point furthest
    from its own centre. For initialisation, k-means++ picks the next centre with
    probability proportional to the squared distance from the nearest existing
    centre, and it is four lines.

### C5: sampling

**The prompt.** "Write greedy, temperature, top-k and top-p sampling from a logit
vector. Then tell me which you would use for a coding assistant."

**Time.** 25 minutes. **Canon #32.**

**Acceptance test.** `pytest tests/test_transformer_gpt.py`. By hand: temperature
approaching zero matches greedy, `top_k=1` matches greedy, and top-p with `p=1.0`
matches plain sampling.

??? tip "Hint 1"
    Temperature divides the logits before the softmax. Lower means sharper.

??? tip "Hint 2"
    Top-k: `kth = logits.topk(k, -1).values[:, -1:]`, then
    `logits.masked_fill(logits < kth, mask_value)`. Masking in logit space keeps
    the renormalisation automatic.

??? tip "Hint 3"
    Top-p: sort descending, `cumsum`, drop everything whose cumulative mass
    *before* it already exceeded `p`, then scatter back to the original vocabulary
    positions. Testing `cum > p` instead of `cum - sorted_p > p` can leave an empty
    set when the top token alone exceeds `p`.

### C6: KV cache

**The prompt.** "Here is a working decoder that recomputes everything each step.
Add a KV cache. Then tell me what it costs in memory."

**Time.** 25 minutes. **Canon #33.**

**Acceptance test.** `pytest tests/test_transformer_multihead.py`. By hand: greedy
generation with and without the cache must produce identical token ids, and the
per-token time must stop growing with the prefix length.

??? tip "Hint 1"
    At step `t` the model runs on one token. Q is `(B, H, 1, dh)`; K and V are the
    concatenated cache, `(B, H, t+1, dh)`.

??? tip "Hint 2"
    Concatenate on the sequence axis, which is `dim=2` in `(B, H, S, dh)` layout.
    Write the layout in a comment at the top of the class, because the other
    common layout puts the sequence at `dim=1`.

??? tip "Hint 3"
    No causal mask is needed during decode: the single query is at the last
    position and every cached key precedes it. Applying a `(1, 1)` causal mask
    anyway is harmless; applying a `(T, T)` one is a shape error. Memory is
    `2 * L * H_kv * dh * T * bytes` per sequence, which at 32 layers, 8 kv heads,
    dh 128, T 8192 in fp16 is 1.07 GB.

### C7: the PPO update

**The prompt.** "Implement the PPO clipped objective and one update step over a
batch of transitions."

**Time.** 25 minutes. **Canon #53.**

**Acceptance test.** `pytest tests/test_rl_ppo.py`. By hand: at `ratio = 1` the
loss equals `-mean(advantage)`; a positive advantage stops contributing gradient
once the ratio exceeds `1 + eps`.

??? tip "Hint 1"
    `ratio = exp(logp_new - logp_old)`, with `logp_old` detached and computed once
    before the inner epochs.

??? tip "Hint 2"
    `loss = -min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A).mean()`. The `min` is
    what makes it pessimistic in both directions, and taking the max instead
    inverts the trust region.

??? tip "Hint 3"
    Three terms in the total loss: the clipped surrogate, a value loss (often
    clipped the same way), and an entropy bonus with a negative coefficient.
    Normalising advantages per batch is standard and worth mentioning; say that it
    is a variance reduction with a small bias.

### C8: average precision

**The prompt.** "Given detections with scores and ground-truth boxes, compute
average precision at IoU 0.5. Then extend it to mAP over classes."

**Time.** 25 minutes. **Canon #59.**

**Acceptance test.** `pytest tests/test_evaluation_detection_map.py`. By hand: a
perfect detector scores 1.0, and a detector that ranks one false positive above
all true positives scores strictly less.

??? tip "Hint 1"
    Sort detections by score descending, then walk down marking each as a true or
    false positive by greedy IoU matching against unmatched ground truth.

??? tip "Hint 2"
    Precision and recall are cumulative down that sorted list:
    `tp.cumsum() / (tp.cumsum() + fp.cumsum())` and `tp.cumsum() / n_gt`.

??? tip "Hint 3"
    The integration convention matters and interviewers ask. COCO interpolates
    precision to the maximum at any higher recall, then averages over 101 recall
    points. Each ground-truth box may be matched once; a second detection of the
    same object is a false positive, which is what NMS is protecting you from.

### C9: BPE

**The prompt.** "Train a byte-pair-encoding tokenizer on a small corpus, then
encode a string with it."

**Time.** 25 minutes. **Canon #24.**

**Acceptance test.** `pytest tests/test_transformer_tokenizers.py`. By hand: a
round trip of encode then decode is the identity, and the number of merges plus the
alphabet size equals the vocabulary size.

??? tip "Hint 1"
    Represent each word as a tuple of symbols. Count adjacent pairs across the
    corpus weighted by word frequency. Merge the most frequent pair. Repeat.

??? tip "Hint 2"
    Store the merges in order. Encoding replays them in the same order on the new
    word, which is what makes encoding deterministic and consistent with training.

??? tip "Hint 3"
    The encode loop picks, at each step, the pair in the current word with the
    lowest merge rank, not the first pair from the left. Getting that wrong gives
    a tokenization that is valid and different from the training-time one, and
    perplexity looks fine because the model was trained on whichever one you used.

## Tier 3: extended, 45 minutes each

Take-home scale, or the whole round when the interviewer wants to see you build
something with parts.

### E1: a GPT forward pass

**The prompt.** "Build a small GPT. Token and position embeddings, N decoder
blocks, a language-model head. Then run one training step on random data and show
me the loss."

**Time.** 45 minutes. **Canon #29, #30, #31.**

**Acceptance test.** `pytest tests/test_transformer_gpt.py`. By hand: the initial
loss is about `log(V)`, the shapes survive a batch of 1 and a sequence of 1, and
editing token `t+1` leaves the logits at positions `0..t` unchanged.

??? tip "Hint 1"
    Pre-norm blocks: `x = x + attn(ln1(x))` then `x = x + mlp(ln2(x))`. Post-norm
    is the original paper and needs a warmup to train.

??? tip "Hint 2"
    The loss needs the shift: `logits[:, :-1]` against `ids[:, 1:]`, flattened to
    `(-1, V)` and `(-1,)` for `F.cross_entropy`.

??? tip "Hint 3"
    The initial loss check is the fastest correctness signal you have. A randomly
    initialised model over a vocabulary of size V should produce a loss of about
    `log V`. If it is much lower, you have a label leak; much higher, your
    initialisation or your normalisation is wrong.

### E2: convolution via im2col

**The prompt.** "Implement a 2-D convolution forward pass in NumPy with stride and
padding. Then the backward."

**Time.** 45 minutes. **Canon #16.**

**Acceptance test.** `pytest tests/test_vision_conv.py`, or compare against
`F.conv2d` on random input, and the backward against `torch.autograd.grad`.

??? tip "Hint 1"
    Output size is `(H + 2p - k) // s + 1`. Work it out for `H=7, k=3, s=2, p=0`
    (the answer is 3) before you write any code.

??? tip "Hint 2"
    im2col makes a `(C*k*k, H_out*W_out)` matrix per image, so the convolution
    becomes `W.reshape(C_out, -1) @ cols`.

??? tip "Hint 3"
    The backward has three parts. The weight gradient is `dY @ cols.T`. The input
    gradient is `col2im(W.T @ dY)`, and `col2im` must *accumulate* where patches
    overlap, so it is `np.add.at` and not an assignment. Stride 1 with a 3x3
    kernel means every interior pixel appears in nine patches.

### E3: a scalar autograd engine

**The prompt.** "Build a tiny autograd engine. A Value class with `+`, `*`, `tanh`,
and a `backward()` that does reverse-mode over the graph."

**Time.** 45 minutes. **Canon #12.**

**Acceptance test.** `pytest tests/test_nn_autograd.py`, or build a small
expression and compare every gradient against `torch`.

??? tip "Hint 1"
    Each `Value` holds `data`, `grad`, the set of its children, and a closure
    `_backward` that pushes its own gradient into its children.

??? tip "Hint 2"
    `backward()` does a topological sort of the graph, sets the output gradient to
    1, then walks the sorted list in reverse calling each `_backward`.

??? tip "Hint 3"
    Gradients *accumulate* (`child.grad += ...`), they do not assign. A node used
    twice in the expression gets contributions from both paths, and assignment
    silently keeps only one. That is the single bug this exercise exists to teach.

### E4: anchors, matching and focal loss

**The prompt.** "Generate anchors over a feature pyramid, match them to ground
truth by IoU, and compute the focal loss for the classification head."

**Time.** 45 minutes. **Canon #20, #21.**

**Acceptance test.** `pytest tests/test_detection_anchors.py tests/test_detection_losses.py`.

??? tip "Hint 1"
    Anchors first, as `(H_f * W_f * A, 4)` with the `A` anchors of one cell
    adjacent. Then a `(n_anchors, n_gt)` IoU matrix, which is the drill you already
    know.

??? tip "Hint 2"
    Matching: an anchor with max IoU above the positive threshold is a positive
    for that ground truth; below the negative threshold it is background; in
    between it is ignored and contributes no loss. Also force the best anchor for
    each ground truth to be positive, so no object goes unmatched.

??? tip "Hint 3"
    Focal loss is `-alpha_t (1 - p_t)^gamma log p_t`. Compute it from logits with
    `F.binary_cross_entropy_with_logits(..., reduction="none")` and multiply by the
    modulating factor. Normalise by the number of *positive* anchors, not by the
    total, or the loss scale moves with the image.

### E5: a reward model

**The prompt.** "Put a scalar head on a small transformer, train it on preference
pairs with the Bradley-Terry loss, and evaluate its pairwise accuracy."

**Time.** 45 minutes. **Canon #54.**

**Acceptance test.** `pytest tests/test_posttrain_reward_model.py`. By hand: the
loss starts at `log 2`, pairwise accuracy on held-out pairs beats 0.5, and adding a
constant to every reward leaves the loss unchanged.

??? tip "Hint 1"
    The head reads the hidden state at the *last real token* of the sequence, which
    with right padding is `h[arange(B), lengths - 1]`, not `h[:, -1]`.

??? tip "Hint 2"
    `loss = -F.logsigmoid(r_chosen - r_rejected).mean()`. Only differences are
    identified, so do not expect the absolute scale to mean anything.

??? tip "Hint 3"
    Two forward passes per step, one for chosen and one for rejected, or one pass
    over a concatenated batch of `2B` and a split afterwards. The second is what
    production code does because it halves the kernel launches.

### E6: a GRPO step

**The prompt.** "Implement one GRPO iteration: sample a group of responses per
prompt, score them with a verifier, compute group-normalised advantages, and take
a clipped policy-gradient step."

**Time.** 45 minutes. **Canon #56, and the shape of #60.**

**Acceptance test.** `pytest tests/test_posttrain_grpo.py tests/test_capstone_components.py`.
By hand: every group's advantages have zero mean, a group with identical rewards
produces exactly zero gradient, and at `ratio = 1` the loss reduces to REINFORCE
with a group baseline.

??? tip "Hint 1"
    Four shapes: rewards `(P, G)`, advantages `(P, G)`, flattened to `(P*G,)` to
    match the flattened batch of responses. The flattening must be prompt-major,
    which means `repeat_interleave` and not `repeat`.

??? tip "Hint 2"
    `A = (r - r.mean(1, keepdim=True)) / (r.std(1, unbiased=False, keepdim=True) + eps)`.
    The group mean is the baseline, which is why GRPO needs no value network.

??? tip "Hint 3"
    Check the fraction of groups whose rewards are all equal before you debug
    anything else. With a binary verifier that fraction is often over half, and
    those groups contribute exactly zero gradient while costing a full generation.
    Dropping them and refilling the batch is DAPO's dynamic sampling.

## The rubric

Score every attempt on five axes. Write the scores down; the pattern across ten
attempts is the study plan.

| Axis | 0 | 1 | 2 |
|---|---|---|---|
| **Correctness** | The acceptance test fails. | Passes the main case, fails an edge case (empty input, batch of one, all-masked row). | Passes, including the edge cases, and you named them before running. |
| **Shapes** | Shape comments missing or wrong. | Present, and you fixed one by running rather than by reasoning. | Every tensor line commented, all correct, written before the line. |
| **Numerics** | A `NaN`, an overflow, or `log(softmax(x))`. | Stable, but you could not say why the trick works. | Stable, and you explained the failure it prevents and at which magnitude. |
| **API fluency** | Stalled more than 15 seconds on an API call, or fought an exception for minutes. | Minor hesitation, recovered without help. | No stalls. Chose `reshape` over `view` deliberately and said why. |
| **Explanation** | Silent, or narrating keystrokes. | Explained after writing. | Stated the plan first, named one trade-off, said what you would test. |

Eight out of ten is a strong hire signal. Six is a hire with a note. Four or below
means the item goes back into rotation.

Two things the rubric deliberately does not score. Speed on its own: finishing in
12 minutes with no explanation is worse than 22 minutes with a good one. And
elegance: a clear loop beats a clever one-liner every time, and interviewers say so
in the debrief.

## Talk while you type

Silence is the most common avoidable loss in a coding round. The interviewer cannot
tell thinking from stuck, and they write down what they observed. Below is the shape
of the narration, drill C1 (multi-head attention) as the worked example. Say it in
your own words; the point is the timing, not the wording.

**Minute 0, before you type anything.**

> "Let me restate it. Multi-head self-attention, batched, causal mask, and it should
> handle padding. I'll do separate Q, K, V projections rather than one fused one,
> because it's easier to read and the interview isn't about the fused kernel.
> Shapes: input is B by T by d, I'll split d into H heads of d over H, do the scores
> per head, mask, softmax, weighted sum, then merge back to B by T by d. Sound
> right?"

Asking "sound right?" is worth two seconds. It catches the case where they wanted
cross-attention.

**Minute 1, the shapes.** Type the comments first, with nothing under them.

> "I'm going to write the shapes down first so I don't lose track. B T d, then
> B T H dh, then B H T dh, scores are B H T S, and back to B T d."

**Minutes 2 to 6, the projections and the split.**

> "Three linears, all d to d. Reshape to B T H dh, then transpose one and two so
> the head axis becomes a batch axis and the matmul at the end is just a batched
> matmul. That transpose makes it non-contiguous, which will matter when I merge
> at the end."

Flagging the contiguity *before* it bites shows you knew, rather than that you
recovered.

**Minutes 6 to 10, the scores and the mask.**

> "Scores are q at k transposed on the last two axes, divided by root dh. Root dh,
> not root d, because the dot product is over the head dimension. The causal mask
> is a lower triangle, T by S, and the padding mask is B by S. I'll unsqueeze the
> padding one to B one one S so it broadcasts over heads and over queries, and
> combine them with an and."

**Minute 10, run it.**

> "Let me run this on B equals 2, T equals 3, d equals 4, two heads, before I go
> further. I want to see B two, H two, T three, S three on the scores."

Running early is graded. It is the difference between an engineer and someone
performing an engineer.

**Minutes 11 to 15, softmax, context, merge.**

> "Softmax over the last axis, the keys. Then weights at v gives B H T dh. Transpose
> back and reshape. Reshape, not view, because of the transpose. Then the output
> projection."

**Minutes 15 to 20, what you would test and what it costs.**

> "Three things I'd test. Rows of the attention weights sum to one. With the causal
> mask, if I change token 3 the logits at positions 0 through 2 don't move. And
> comparing against `scaled_dot_product_attention` on random inputs. On cost: the
> scores are B H T S, so memory is quadratic in sequence length, which is what
> FlashAttention removes by never materialising them. FLOPs are about four B T
> squared d for the two matmuls plus four B T d squared for the projections, so
> projections dominate until T is about d."

**On the mask value, if they ask or if you have time.**

> "I'm using `torch.finfo(dtype).min` rather than negative infinity. If a row is
> entirely padding, every key is masked, and softmax of all negative infinity is
> NaN. A finite value gives a uniform row instead, which the loss mask throws away.
> And minus 1e9 stops being finite in fp16, so `finfo` picks the right constant."

Four sentences that say you have shipped this.

### What to narrate in general

* **Restate the problem** and get agreement. Ten seconds, catches the wrong problem.
* **State the plan** in shapes, before code.
* **Name the choice** when you make one. "Separate projections for readability."
  "Pre-norm because it trains without warmup."
* **Flag a known gotcha** when you reach it, not after.
* **Say when you are about to run** and what you expect to see.
* **Close with the tests** you would write and one cost sentence.

### What not to narrate

Keystrokes ("now I'm typing a for loop"). Apologies ("sorry, this is messy").
Self-assessment ("I'm probably doing this wrong"). Every one of them costs you and
none of them buys anything.

## When you get stuck

Being stuck is expected. Being stuck *silently* is what loses the round. A ladder,
in order, with times.

**0 to 30 seconds: say what you are stuck on.** "I'm trying to remember whether
gather wants the index at the same rank as the source." Half the time the
interviewer just tells you, at no cost, because they want to see the algorithm and
not your API recall.

**30 to 60 seconds: write the shapes of what you have and what you want.** Stuck is
usually a missing transform between two known shapes. Written down, the transform
is often obvious.

**1 to 2 minutes: take the ugly path.** A Python loop over the batch. An explicit
index with `arange`. Two extra intermediate variables. Say out loud that it is the
ugly path and what the vectorised version would be, then keep going. A finished ugly
solution with a stated improvement beats an unfinished elegant one every time.

**2 to 3 minutes: shrink the problem.** Drop the batch axis. One head. Sequence
length 2. Solve that, on paper if needed, then generalise. This is also the honest
way to handle "I've never implemented HNSW": solve the one-layer case and say what
the multi-layer version adds.

**3 minutes: ask for the hint.** "Can you point me at the right primitive here? I
know what I want to compute." Asking at minute 3 costs a little. Grinding until
minute 20 costs the round. Interviewers are explicitly told to give hints and they
grade what you do with them.

Two failure modes that are worse than being stuck. Rewriting from scratch when
something does not work: fix the line, do not restart the file. And debugging by
staring: print the shapes, on a tiny input, at the line before the error.

If you finish early, do not sit. Write the tests you named. Add the edge case. Say
"given another ten minutes I'd add the KV cache, here's where it would go". Filling
the last five minutes with the extension is free signal.

## Three 60-minute mock sessions

Run each end to end without pausing, out loud, ideally to another person. Grade
with the rubric afterwards, not during.

### Mock 1: the perception loop

The shape of an autonomy-company coding round.

| Minutes | What happens |
|---|---|
| 0 to 5 | Warm-up question with no code: "walk me through how a single-stage detector turns a feature map into boxes." |
| 5 to 20 | **W1**, pairwise IoU. Expect a follow-up on GIoU and why IoU has no gradient for disjoint boxes. |
| 20 to 45 | **C2**, NMS, then the per-class extension. Expect "what would you do differently at 10,000 boxes?" |
| 45 to 55 | **C8**, average precision, verbally: the algorithm and the integration convention, no code. |
| 55 to 60 | Your questions. |

What is being graded: the IoU broadcast, the clamp at zero, whether you kept indices
rather than mutating boxes, and whether you know that NMS is `O(n^2)` in the worst
case and what the alternatives are.

### Mock 2: the LLM loop

The shape of a frontier-lab or LLM-infrastructure round.

| Minutes | What happens |
|---|---|
| 0 to 5 | "Draw me the data flow through one decoder layer, with shapes." |
| 5 to 30 | **C1**, multi-head attention with causal and padding masks. |
| 30 to 45 | **C6**, add a KV cache to it. Expect the memory arithmetic. |
| 45 to 55 | **W6**, the DPO loss, cold. Expect "why `logsigmoid`" and "what do you watch during training". |
| 55 to 60 | Your questions. |

What is being graded: the head split and merge without hesitation, `sqrt(dh)`, the
mask broadcast, whether the cached and uncached generations produce identical
tokens, and whether the KV-cache memory formula comes out without a calculator.

### Mock 3: the generalist loop

The shape of a consumer-ranking or platform round, where breadth is the point.

| Minutes | What happens |
|---|---|
| 0 to 15 | **C4**, K-means, with initialisation and the empty-cluster question. |
| 15 to 30 | **C3**, the two-layer MLP backward, with a numerical gradient check. |
| 30 to 45 | **W2** then **W4**, stable softmax and cross-entropy, then LayerNorm forward. |
| 45 to 55 | Verbal: "you have 50 million items and need the top 100 nearest neighbours in 10 ms. What do you build?" |
| 55 to 60 | Your questions. |

What is being graded: whether you reach for the vectorised distance identity
without being asked, whether your backward's `1/N` appears exactly once, whether you
subtract the max before exponentiating without being prompted, and whether the
retrieval answer names IVF, HNSW and product quantisation with a reason for each.

### After a mock

Three lines in the miss log, no more. What you got wrong, which hint level you
needed, and one sentence you wish you had said. After ten mocks the log has maybe
fifteen recurring entries and those fifteen are the whole of your remaining study
list.

## Retype by hand

The five that appear in more loops than anything else. About 80 minutes.

- [ ] **W3** then **C1**: attention, single-head then multi-head with masks, against `F.scaled_dot_product_attention`.
- [ ] **W1** then **C2**: IoU and NMS, against `torchvision.ops.nms`.
- [ ] **C3**: the two-layer MLP backward, with a finite-difference check.
- [ ] **C4**: K-means, asserting the objective is monotone.
- [ ] **W6**: the DPO loss, asserting `log 2` at initialisation.

## References

* Andrej Karpathy, "A Recipe for Training Neural Networks" (2019), on running small and early: [karpathy.github.io/2019/04/25/recipe](http://karpathy.github.io/2019/04/25/recipe/).
* Andrej Karpathy, "micrograd", for the scalar autograd engine in E3: [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd).
* Andrej Karpathy, "nanoGPT", for the GPT structure in E1: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
* Sebastian Raschka, "LLMs-from-scratch", for the same material at a slower pace: [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).
* OpenAI, "Spinning Up in Deep RL", for the PPO and GAE implementations in W7 and C7: [spinningup.openai.com](https://spinningup.openai.com/en/latest/).
