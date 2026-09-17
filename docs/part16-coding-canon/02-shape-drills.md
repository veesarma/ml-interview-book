# Shape and broadcasting drills

> **Why this matters at staff level.** Most coding-round failures are not algorithm
> failures. The candidate knows what attention is and loses eleven minutes to a
> transposed `K`, a mask that broadcast over the wrong axis, or a `mean` that
> averaged over the batch when it should have averaged over the sequence. These
> thirty-two drills are the specific shapes that go wrong, isolated so you can fix
> them once.

## TL;DR: the interview card

* Broadcasting aligns shapes from the **right**. A missing leading axis is inserted as 1; an existing axis of size 1 stretches; anything else is an error.
* `(N,) + (N, 1)` gives `(N, N)`. This is the most expensive typo in the book.
* Write the output shape as a comment *before* the line that produces it.
* A mask enters as an additive bias of shape `(B, 1, T, S)`, broadcast over heads, added to the scores before the softmax.
* Use a finite large negative mask value, not `-inf`: a fully masked query row gives `NaN` out of softmax.
* Per-sample reductions divide by the mask sum for that sample. `x.mean()` over a padded batch is wrong by however much padding you have.
* `gather` needs the index at the same rank as the source, and you squeeze the gathered axis afterwards.
* `reshape` after `transpose`, never `view`.

## How to use this chapter

Read the prompt, say the answer out loud with every intermediate shape, then open
the solution. If your spoken answer had the right shapes but a different route,
that is a pass. If you named a shape you could not justify, redo it.

Ten drills, out loud, is the standard warm-up before a coding round. The study plan
puts a set of them in week 1 and repeats them from week 4 onward.

A few drills share a dimension vocabulary with the rest of the book: `B` batch, `T`
query length, `S` key length, `d` model width, `H` heads, `dh` head dimension, `V`
vocabulary, `N`/`M` counts of boxes or points, `C,H_img,W` image dimensions, `K`
classes, `E` experts, `A` anchors, `P` patch side.

## Part 1: broadcasting

### Drill 1: the rule itself

`a` has shape `(3, 1, 5)` and `b` has shape `(4, 5)`. What does `a + b` give, and
what would `a * b.T` give?

??? success "Answer"
    ```text
    a       (3, 1, 5)
    b          (4, 5)   ->  right-aligned, a leading 1 is inserted: (1, 4, 5)
    a + b   (3, 4, 5)
    ```

    `b.T` has shape `(5, 4)`. Right-aligning `(3, 1, 5)` against `(5, 4)` compares
    5 with 4 on the last axis, and neither is 1, so it raises.

    **The trap.** Alignment is from the right, always. People reason from the left
    because that is how they read the shape tuple.

### Drill 2: the expensive typo

`scores` has shape `(N,)` and `bias` has shape `(N, 1)`. What is `scores + bias`,
and how much memory does it use at N = 100000 in float32?

??? success "Answer"
    ```text
    scores        (N,)  ->  (1, N)
    bias       (N, 1)
    sum        (N, N)
    ```

    At N = 100000 that is 10^10 elements, 40 GB. The allocation fails and the
    traceback points at the addition, not at the missing `[:, 0]` three lines
    earlier.

    **The trap.** Any time a `(N, 1)` meets a `(N,)`, you get an outer operation.
    Keeping shape comments on both operands is what catches it. The habit that
    prevents it is to squeeze the moment you are done with a `keepdims` reduction.

### Drill 3: pairwise squared distances

Given `A` of shape `(N, d)` and `B` of shape `(M, d)`, produce the matrix of
squared Euclidean distances with shape `(N, M)`, without a loop and without
forming an `(N, M, d)` intermediate.

??? success "Answer"
    ```python
    a2 = (A ** 2).sum(axis=1, keepdims=True)        # (N, 1)
    b2 = (B ** 2).sum(axis=1)                       # (M,)
    d2 = a2 - 2.0 * (A @ B.T) + b2                  # (N, 1) + (N, M) + (M,) -> (N, M)
    d2 = np.maximum(d2, 0.0)                        # (N, M)
    ```

    The expansion is $\|a - b\|^2 = \|a\|^2 - 2 a^\top b + \|b\|^2$. The
    `keepdims=True` on the first term and its absence on the second is what makes
    the broadcast land on the right axes.

    **The trap.** Two of them. The naive `((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)`
    is correct and allocates `N * M * d` floats, which is 400x more memory at
    d = 100. And the expanded form can produce small negative values from
    floating-point cancellation, so a `sqrt` afterwards returns `NaN` unless you
    clamp at zero first. Canon #4 and #5 both depend on this.

### Drill 4: which axis did the reduction eat

`x` has shape `(B, T, d)`. Give the shapes of `x.mean(1)`, `x.mean(1, keepdims=True)`,
`x.mean((0, 1))` and `x.mean()`. Which one do you want for LayerNorm?

??? success "Answer"
    ```text
    x.mean(1)                  (B, d)
    x.mean(1, keepdims=True)   (B, 1, d)
    x.mean((0, 1))             (d,)
    x.mean()                   ()          a scalar
    ```

    LayerNorm normalises over the feature axis per token, so it wants
    `x.mean(-1, keepdims=True)` of shape `(B, T, 1)`. BatchNorm over a sequence
    normalises per feature across batch and time, so it wants `x.mean((0, 1))` of
    shape `(d,)`.

    **The trap.** Both are "the mean", both run without error, and the difference
    between them is the difference between LayerNorm and BatchNorm. Canon #13 and
    #14 are the same ten lines with this axis changed.

## Part 2: attention

### Drill 5: the head split

`x` has shape `(B, T, d)` with `H` heads and `d = H * dh`. Produce `(B, H, T, dh)`.
Then go back to `(B, T, d)`. Why can the return trip not use `view`?

??? success "Answer"
    ```python
    B, T, d = x.shape                                   # (B, T, d)
    q = x.reshape(B, T, H, dh)                          # (B, T, H, dh)
    q = q.transpose(1, 2)                               # (B, H, T, dh)
    # ...
    out = q.transpose(1, 2)                             # (B, T, H, dh)
    out = out.reshape(B, T, d)                          # (B, T, d)
    ```

    `transpose` swaps strides and leaves the buffer alone, so the result is not
    contiguous and `view` raises. `reshape` notices and makes the copy.
    `out.contiguous().view(B, T, d)` is the same thing said explicitly.

    **The trap.** Reshaping straight from `(B, H, T, dh)` to `(B, T, d)` without
    the transpose back also produces a `(B, T, d)` tensor when `H * T == T * H`,
    so the shape check passes and the heads come out interleaved with the
    positions. The loss still falls. The model is just worse than it should be,
    and nothing tells you.

### Drill 6: the flagship

Given `Q` of shape `(B, T, d)`, `K` and `V` of shape `(B, S, d)`, a padding mask
of shape `(B, S)` where 1 means "real token", and `H` heads, produce the attention
output of shape `(B, T, d)` with the mask applied correctly.

??? success "Answer"
    ```python
    B, T, d = Q.shape                                        # (B, T, d)
    S = K.shape[1]
    dh = d // H

    q = Q.reshape(B, T, H, dh).transpose(1, 2)               # (B, H, T, dh)
    k = K.reshape(B, S, H, dh).transpose(1, 2)               # (B, H, S, dh)
    v = V.reshape(B, S, H, dh).transpose(1, 2)               # (B, H, S, dh)

    scores = q @ k.transpose(-2, -1) / dh ** 0.5             # (B, H, T, S)

    keep = pad_mask.bool()[:, None, None, :]                 # (B, 1, 1, S)
    scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)  # (B, H, T, S)

    w = torch.softmax(scores, dim=-1)                        # (B, H, T, S)
    ctx = w @ v                                              # (B, H, T, dh)
    out = ctx.transpose(1, 2).reshape(B, T, d)               # (B, T, d)
    ```

    Four shape facts carry the whole thing. The scale is `sqrt(dh)`, the head
    dimension, not `sqrt(d)`. The transpose on `k` is on the last two axes, which
    `-2, -1` says independently of rank. The mask is unsqueezed to `(B, 1, 1, S)`
    so it broadcasts over heads *and* over query positions, because a padded key
    is invisible to every query. The softmax runs over `dim=-1`, the key axis.

    **The trap.** Scaling by `sqrt(d)` instead of `sqrt(dh)` is the most common
    single error, and with H = 8 it makes the logits 2.8x too small, so attention
    comes out nearly uniform and the model trains slowly with no error anywhere.

### Drill 7: causal and padding together

Add causal masking to drill 6, for the case `S == T`. Produce the combined mask and
say what shape it has.

??? success "Answer"
    ```python
    causal = torch.tril(torch.ones(T, S, dtype=torch.bool, device=Q.device))  # (T, S)
    keep_pad = pad_mask.bool()[:, None, None, :]                              # (B, 1, 1, S)
    keep = causal[None, None, :, :] & keep_pad                                # (B, 1, T, S)
    bias = torch.zeros(keep.shape, dtype=scores.dtype, device=Q.device)       # (B, 1, T, S)
    bias = bias.masked_fill(~keep, torch.finfo(scores.dtype).min)             # (B, 1, T, S)
    scores = scores + bias                                                    # (B, H, T, S)
    ```

    The causal part depends on the query position and not on the batch, so it is
    `(T, S)`. The padding part depends on the batch and not on the query position,
    so it is `(B, 1, 1, S)`. Their `and` is `(B, 1, T, S)`, and the head axis of 1
    broadcasts across `H` for free.

    **The trap.** Building the combined mask at `(B, H, T, S)` costs `H` times the
    memory for no benefit. At B = 8, H = 32, T = 4096 in float32 that is 16 GB
    instead of 512 MB.

### Drill 8: why not `-inf`

In drill 7, the mask value is `torch.finfo(dtype).min` and not `float("-inf")`.
Give the two reasons.

??? success "Answer"
    A right-padded batch has query rows that are themselves padding. Every key in
    such a row is masked, so the row of scores is uniformly `-inf`, and
    `softmax` computes `exp(-inf - (-inf)) = exp(nan)`. The `NaN` then flows into
    the output, and because `NaN * 0 == NaN` it survives every subsequent loss
    mask and poisons the entire backward pass. A finite value gives a harmless
    uniform row instead, and the loss mask discards it.

    Second, `-1e9` in float16 is below `-65504` and rounds to `-inf`, so a mask
    value that was finite in float32 stops being finite the moment you turn on
    mixed precision. `torch.finfo(scores.dtype).min` picks `-65504` under fp16 and
    `-3.4e38` under fp32.

    **The trap.** This bug appears only with right padding and only when a whole
    sequence in the batch is short. It passes every test written on a uniform-length
    batch.

### Drill 9: cross-attention

In cross-attention, `Q` comes from the decoder and `K`, `V` from the encoder. With
decoder length `T = 7` and encoder length `S = 40`, give every intermediate shape
and say which mask applies.

??? success "Answer"
    ```text
    q      (B, H,  7, dh)
    k      (B, H, 40, dh)
    v      (B, H, 40, dh)
    scores (B, H,  7, 40)
    w      (B, H,  7, 40)     softmax over the last axis, 40 keys
    ctx    (B, H,  7, dh)
    out    (B,  7, d)
    ```

    Only the encoder's padding mask applies, at `(B, 1, 1, 40)`. There is no causal
    mask: a decoder position may attend to the whole source. The decoder's own
    self-attention, a separate sub-layer, carries the causal mask.

    **The trap.** Reusing the `(T, T)` causal mask here. It is the wrong shape and,
    when T happens to equal S, it is the wrong semantics and runs anyway.

### Drill 10: GQA head repetition

`q` is `(B, H, T, dh)` with H = 32. `k` and `v` are `(B, H_kv, S, dh)` with
H_kv = 8. Produce `(B, H, T, S)` scores.

??? success "Answer"
    ```python
    group = H // H_kv                                        # 4 query heads per kv head
    k = k[:, :, None, :, :]                                  # (B, H_kv, 1, S, dh)
    k = k.expand(B, H_kv, group, S, dh)                      # (B, H_kv, group, S, dh)
    k = k.reshape(B, H, S, dh)                               # (B, H, S, dh)
    scores = q @ k.transpose(-2, -1)                         # (B, H, T, S)
    ```

    `expand` costs no memory; the `reshape` after it does copy, which is why
    production kernels fuse the repetition into the matmul instead.

    **The trap.** The grouping has to be `repeat_interleave` semantics, so query
    heads 0 to 3 share kv head 0. Using `k.repeat(1, group, 1, 1)` tiles instead,
    giving heads 0, 8, 16, 24 the same kv head. The shapes match. The model is
    wrong, and after a weight conversion from a real checkpoint it is wrong in a
    way that looks like a bad checkpoint.

### Drill 11: KV cache concatenation

At decode step `t`, the model runs on one new token. The cache holds `k_cache` of
shape `(B, H, t, dh)`. Give the shapes through one step, and the total cache bytes
for L = 32 layers, H_kv = 8, dh = 128, T = 8192, B = 1 in fp16.

??? success "Answer"
    ```python
    k_new = k_proj(x_t).reshape(B, 1, H, dh).transpose(1, 2)  # (B, H, 1, dh)
    k_cache = torch.cat([k_cache, k_new], dim=2)              # (B, H, t+1, dh)
    scores = q_new @ k_cache.transpose(-2, -1)                # (B, H, 1, t+1)
    ctx = torch.softmax(scores, -1) @ v_cache                 # (B, H, 1, dh)
    ```

    Bytes: `2 (K and V) * L * H_kv * dh * T * 2 bytes` =
    `2 * 32 * 8 * 128 * 8192 * 2` = 1.07 GB for one sequence.

    **The trap.** The concat is on `dim=2`, the sequence axis of a `(B, H, S, dh)`
    layout. It is `dim=1` if you kept the cache in `(B, S, H, dh)` layout. Pick one
    layout and put it in a comment at the top of the class, because the two conventions
    are both common and both look right.

## Part 3: reductions over ragged data

### Drill 12: masked mean

`x` is `(B, T, d)` and `mask` is `(B, T)` with 1 on real tokens. Produce the mean
over real tokens, once per sample, shape `(B, d)`.

??? success "Answer"
    ```python
    m = mask[:, :, None].to(x.dtype)                 # (B, T, 1)
    total = (x * m).sum(dim=1)                       # (B, d)
    count = m.sum(dim=1).clamp_min(1.0)              # (B, 1)
    out = total / count                              # (B, d)
    ```

    **The trap.** `x.mean(dim=1)` divides by `T` for every sample, including the
    one that is 90% padding. On a batch sorted by length, the error correlates with
    position in the batch, which makes it look like a data bug.

### Drill 13: per-sample versus per-batch loss

`logits` is `(B, T, V)`, `ids` is `(B, T)`, `loss_mask` is `(B, T)`. Give the
token-mean loss and the sequence-mean loss, and say which one SFT uses.

??? success "Answer"
    ```python
    flat = F.cross_entropy(
        logits.reshape(-1, V), ids.reshape(-1), reduction="none"
    ).reshape(ids.shape)                             # (B, T)

    token_mean = (flat * loss_mask).sum() / loss_mask.sum()          # ()
    per_seq = (flat * loss_mask).sum(1) / loss_mask.sum(1).clamp_min(1)  # (B,)
    seq_mean = per_seq.mean()                                        # ()
    ```

    SFT normally uses the token mean, so a long answer contributes more gradient
    than a short one. The sequence mean weights every example equally regardless
    of length, which is what you want when answer length is confounded with
    quality. GRPO's length bias is exactly this choice made at the response level.

    **The trap.** They differ whenever lengths differ, and both are defensible, so
    an interviewer asking "which did you use and why" is asking a real question.

### Drill 14: ragged to padded

You have a list of `B` sequences of token ids with different lengths. Produce
`ids` of shape `(B, T_max)`, an attention mask, and the index of the last real
token per row.

??? success "Answer"
    ```python
    T = max(len(s) for s in seqs)
    ids = np.full((B, T), PAD, dtype=np.int64)       # (B, T)
    mask = np.zeros((B, T), dtype=np.int64)          # (B, T)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    last = mask.sum(axis=1) - 1                      # (B,) index of the final real token
    ```

    To read the final hidden state per row:
    `h[torch.arange(B), torch.as_tensor(last)]`, shape `(B, d)`.

    **The trap.** With right padding, `h[:, -1]` is the padding position for every
    short row. Reward models read the last token, so a reward model with right
    padding and `h[:, -1]` scores the pad embedding, which is a constant, and it
    trains to a Bradley-Terry loss of exactly `log 2` and stays there.

### Drill 15: cumsum for packing

You pack several short sequences into one row of length `T` and keep `seg` of
shape `(T,)` giving each token's segment id. Produce the block-diagonal attention
mask of shape `(T, T)` that stops one packed sequence attending to another.

??? success "Answer"
    ```python
    same = seg[:, None] == seg[None, :]              # (T, T) True inside a segment
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))  # (T, T)
    keep = same & causal                             # (T, T)
    ```

    To build `seg` itself from a list of lengths:

    ```python
    starts = torch.cumsum(torch.as_tensor(lengths), 0) - torch.as_tensor(lengths)  # (n_seq,)
    seg = torch.zeros(T, dtype=torch.long)           # (T,)
    seg[starts] = 1
    seg = torch.cumsum(seg, 0) - 1                   # (T,) 0, 0, 1, 1, 1, 2, ...
    ```

    **The trap.** Packing without the segment mask is a silent data leak: the model
    conditions on the end of the previous document. It improves perplexity on the
    packed training set and does nothing for the model.

### Drill 16: cumsum for top-p

`probs` is `(B, V)`. Keep the smallest set of tokens whose cumulative probability
reaches `p`, and zero the rest.

??? success "Answer"
    ```python
    sorted_p, idx = torch.sort(probs, dim=-1, descending=True)   # (B, V) each
    cum = torch.cumsum(sorted_p, dim=-1)                         # (B, V)
    drop = cum - sorted_p > p                                    # (B, V)
    sorted_p = sorted_p.masked_fill(drop, 0.0)                   # (B, V)
    out = torch.zeros_like(probs).scatter_(-1, idx, sorted_p)    # (B, V)
    out = out / out.sum(-1, keepdim=True)                        # (B, V)
    ```

    `cum - sorted_p` is the cumulative mass *before* each token, so the comparison
    keeps the token that crosses the threshold. Testing `cum > p` drops it and can
    leave an empty set when the top token already exceeds `p`.

    **The trap.** The `scatter_` at the end puts the kept probabilities back at
    their original vocabulary positions. Skipping it and sampling from `sorted_p`
    gives you an index into the sorted order, not a token id.

## Part 4: gather and scatter

### Drill 17: token log-probabilities

`logits` is `(B, T, V)` and `ids` is `(B, T)`. Produce `(B, T)` log-probabilities
of the realised tokens.

??? success "Answer"
    ```python
    logp = F.log_softmax(logits, dim=-1)                  # (B, T, V)
    out = logp.gather(-1, ids[:, :, None])                # (B, T, 1)
    out = out[:, :, 0]                                    # (B, T)
    ```

    NumPy: `np.take_along_axis(logp, ids[:, :, None], axis=-1)[:, :, 0]`.

    **The trap.** `torch.gather` requires the index to have the same rank as the
    source and every non-gathered axis to match exactly. NumPy broadcasts the index
    instead, so a NumPy implementation ported to Torch by find-and-replace throws a
    shape error, and a Torch implementation ported to NumPy silently produces a
    different answer when the shapes happen to broadcast.

### Drill 18: the label shift

For a causal LM, `logits` is `(B, T, V)` and `ids` is `(B, T)`. Give the loss.

??? success "Answer"
    ```python
    pred = logits[:, :-1, :]                              # (B, T-1, V)
    target = ids[:, 1:]                                   # (B, T-1)
    loss = F.cross_entropy(pred.reshape(-1, V), target.reshape(-1))
    ```

    `logits[:, i]` is the distribution over the token at position `i + 1`, so the
    prediction at `i` pairs with the id at `i + 1`. Both slices lose one position,
    and it is the same position at opposite ends.

    **The trap.** With a visual prefix of `N_q` tokens in front of the text, the
    logit that scores text token `t` sits at absolute index `N_q + t - 1`, so the
    slice is `logits[:, N_q - 1 : N_q - 1 + T]`. Canon #60 does exactly this.
    Getting it wrong shifts the loss by one token, and the model trains happily to
    predict the wrong thing.

### Drill 19: top-k expert routing

`x` is `(B, T, d)`, a router gives `(B, T, E)` logits, and you keep the top `k`
experts per token. Produce the gates and the flat token index per expert.

??? success "Answer"
    ```python
    logits = router(x)                                    # (B, T, E)
    top_val, top_idx = logits.topk(k, dim=-1)             # (B, T, k) each
    gate = torch.softmax(top_val, dim=-1)                 # (B, T, k)

    flat_x = x.reshape(B * T, d)                          # (BT, d)
    flat_idx = top_idx.reshape(B * T, k)                  # (BT, k)
    out = torch.zeros_like(flat_x)                        # (BT, d)
    for e in range(E):
        rows, slot = (flat_idx == e).nonzero(as_tuple=True)   # (n_e,) each
        if rows.numel() == 0:
            continue
        y = experts[e](flat_x[rows])                          # (n_e, d)
        out.index_add_(0, rows, y * gate.reshape(B * T, k)[rows, slot][:, None])
    ```

    The softmax runs over the `k` kept logits, not over all `E`, so the gates sum
    to one across the experts actually used.

    **The trap.** `out[rows] += ...` instead of `index_add_` drops every
    contribution but the last when a token picked two experts, which is exactly
    the k > 1 case you built this for. Canon #37.

### Drill 20: scatter for one-hot and for counting

Produce a one-hot matrix `(N, K)` from labels `(N,)` two ways, and a class-count
vector `(K,)`.

??? success "Answer"
    ```python
    oh = F.one_hot(labels, K).float()                     # (N, K)
    oh = torch.zeros(N, K).scatter_(1, labels[:, None], 1.0)   # (N, K)
    counts = torch.bincount(labels, minlength=K)          # (K,)
    counts = torch.zeros(K).index_add_(0, labels, torch.ones(N))  # (K,)
    ```

    NumPy: `np.eye(K)[labels]` and `np.bincount(labels, minlength=K)`.

    **The trap.** `counts[labels] += 1` gives 1 for every class that appears at
    least once, in both libraries. Indexed assignment reads, adds and writes once,
    so duplicate indices overwrite each other.

### Drill 21: class-aware NMS in one call

You have boxes `(N, 4)`, scores `(N,)` and class ids `(N,)`, and you want NMS run
independently per class using a single global NMS. How?

??? success "Answer"
    ```python
    span = boxes.max() - boxes.min() + 1                  # scalar
    offsets = cls.to(boxes.dtype)[:, None] * span         # (N, 1)
    shifted = boxes + offsets                             # (N, 4)
    keep = nms(shifted, scores, iou_threshold)            # (n_keep,)
    ```

    Adding a per-class offset larger than the coordinate range moves each class to
    a disjoint region of the plane, so boxes of different classes can never
    overlap, so a single NMS behaves per class.

    **The trap.** The offset is added to all four coordinates, not two. Adding it
    to `x1, x2` only shifts horizontally and two classes can still collide
    vertically. This is `torchvision.ops.batched_nms` and it is worth being able to
    derive on the spot, because the naive alternative is a Python loop over classes.

## Part 5: images, patches and boxes

### Drill 22: image to patch tokens

`images` is `(B, C, H_img, W)` with `H_img = W = 224` and patch side `P = 16`.
Produce patch tokens `(B, N, P*P*C)` and say what `N` is.

??? success "Answer"
    ```python
    G = H_img // P                                        # 14 patches per side
    x = images.reshape(B, C, G, P, G, P)                  # (B, C, G, P, G, P)
    x = x.permute(0, 2, 4, 1, 3, 5)                       # (B, G, G, C, P, P)
    x = x.reshape(B, G * G, C * P * P)                    # (B, 196, 768)
    ```

    `N = (H_img / P) * (W / P) = 196`. The permute is what puts the two grid axes
    first and the three content axes last, so the final reshape groups the right
    things.

    **The trap.** Going straight from `(B, C, G, P, G, P)` to `(B, G*G, C*P*P)`
    without the permute produces the right shape and the wrong contents: each
    "patch" is a horizontal strip of the image rather than a square. The ViT still
    trains, a few points worse, and nothing errors. Canon #39.

### Drill 23: patches back to an image

Invert drill 22.

??? success "Answer"
    ```python
    x = tokens.reshape(B, G, G, C, P, P)                  # (B, G, G, C, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)                       # (B, C, G, P, G, P)
    images = x.reshape(B, C, G * P, G * P)                # (B, C, H, W)
    ```

    The inverse permutation of `(0, 2, 4, 1, 3, 5)` is `(0, 3, 1, 4, 2, 5)`. Read
    it as: the axis that ended up at position 1 came from position 3, and so on.

    **The trap.** Getting the inverse permutation wrong is easy and the symptom is
    an image that looks like a shuffled jigsaw, which at least tells you
    immediately. Canon #43 and #44 need this for image decoders.

### Drill 24: im2col

`x` is `(B, C, H_img, W)`, kernel `k`, stride `s`, padding `p`. Produce the im2col
matrix and give the output spatial size.

??? success "Answer"
    ```python
    H_out = (H_img + 2 * p - k) // s + 1
    W_out = (W + 2 * p - k) // s + 1
    cols = F.unfold(x, kernel_size=k, stride=s, padding=p)   # (B, C*k*k, H_out*W_out)
    w = weight.reshape(C_out, C * k * k)                     # (C_out, C*k*k)
    out = w @ cols                                           # (B, C_out, H_out*W_out)
    out = out.reshape(B, C_out, H_out, W_out)                # (B, C_out, H_out, W_out)
    ```

    `F.unfold` puts the patch content on axis 1 and the spatial position on axis 2,
    which is the transpose of how most textbook im2col diagrams draw it.

    **The trap.** The output-size formula uses floor division, so a stride that
    does not divide evenly silently drops the last column of the input. Working it
    out on paper for one concrete case (`H=7, k=3, s=2, p=0` gives 3) is faster
    than rederiving it under pressure. Canon #16.

### Drill 25: pairwise IoU

`a` is `(N, 4)` and `b` is `(M, 4)`, both in `(x1, y1, x2, y2)`. Produce `(N, M)`
IoU.

??? success "Answer"
    ```python
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)   # (N,)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)   # (M,)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])       # (N, M, 2) top-left
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])       # (N, M, 2) bottom-right
    wh = (rb - lt).clip(0)                                # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]                       # (N, M)

    union = area_a[:, None] + area_b[None, :] - inter     # (N, M)
    iou = inter / np.maximum(union, 1e-9)                 # (N, M)
    ```

    **The trap.** The `clip(0)` on `wh` is what handles boxes that do not overlap
    at all. Without it, two disjoint boxes give a negative width times a negative
    height, so a positive intersection area, and IoU comes out large for boxes on
    opposite sides of the image. Canon #18, and the same `(N, M)` pattern is the
    cost matrix for Hungarian matching in #40.

### Drill 26: anchors over a feature grid

A feature map is `(H_f, W_f)` with stride 16 and `A` anchor shapes given as
`(A, 2)` widths and heights. Produce all anchors as `(H_f * W_f * A, 4)` in
`(x1, y1, x2, y2)`, ordered so that the `A` anchors of one cell are adjacent.

??? success "Answer"
    ```python
    ys = (np.arange(H_f) + 0.5) * 16                      # (H_f,)
    xs = (np.arange(W_f) + 0.5) * 16                      # (W_f,)
    cy, cx = np.meshgrid(ys, xs, indexing="ij")           # (H_f, W_f) each
    centres = np.stack([cx, cy], axis=-1)                 # (H_f, W_f, 2)

    half = wh / 2.0                                       # (A, 2)
    c = centres[:, :, None, :]                            # (H_f, W_f, 1, 2)
    boxes = np.concatenate([c - half, c + half], axis=-1) # (H_f, W_f, A, 4)
    boxes = boxes.reshape(-1, 4)                          # (H_f*W_f*A, 4)
    ```

    `indexing="ij"` makes the first output vary along rows. With the default
    `"xy"` the two outputs come back swapped, which transposes the whole anchor
    grid.

    **The trap.** The ordering promise. `reshape(-1, 4)` on `(H_f, W_f, A, 4)`
    gives cell-major, anchor-minor, which is what a detection head with `A * 4`
    output channels per cell expects if it is reshaped as `(H_f, W_f, A, 4)`. If
    the head instead outputs `(A * 4, H_f, W_f)` and you permute to
    `(H_f, W_f, A, 4)`, you must check whether the channel axis was
    anchor-major or coordinate-major. Getting it backwards makes every box a
    mixture of two anchors. Canon #20.

### Drill 27: BEV scatter

Points are `(P, 3)` in metres. The BEV grid covers x in `[-50, 50]` and y in
`[-50, 50]` at 0.5 m resolution. Scatter per-point features `(P, C)` into a BEV
feature map `(C, X, Y)` by summing.

??? success "Answer"
    ```python
    res, lo = 0.5, -50.0
    X = Y = int(100 / res)                                # 200
    ix = ((pts[:, 0] - lo) / res).long()                  # (P,)
    iy = ((pts[:, 1] - lo) / res).long()                  # (P,)
    valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)   # (P,)
    flat = ix[valid] * Y + iy[valid]                      # (P_valid,)

    bev = torch.zeros(C, X * Y)                           # (C, X*Y)
    bev.index_add_(1, flat, feats[valid].T)               # (C, X*Y)
    bev = bev.reshape(C, X, Y)                            # (C, X, Y)
    ```

    **The trap.** Two. The validity mask has to come before the flattening, because
    a point at x = -60 gives `ix = -20`, and negative indices wrap round to the far
    edge of the grid instead of erroring. And `index_add_` is mandatory: many
    points land in the same cell, which is the entire point of a BEV pooling layer.

## Part 6: einsum and the rest

### Drill 28: read an einsum string

Explain `torch.einsum("bthd,bshd->bhts", q, k)` term by term, and give the
equivalent written with `transpose` and `@`.

??? success "Answer"
    Indices: `b` batch, `t` query position, `s` key position, `h` head, `d` head
    dimension.

    `d` appears in both inputs and not in the output, so it is summed over: it is
    the contracted axis. `b` and `h` appear in both inputs and in the output, so
    they are batch axes. `t` appears only in the first input and `s` only in the
    second, so they become the two free axes of the result.

    ```python
    qs = q.transpose(1, 2)                                # (B, H, T, dh)
    ks = k.transpose(1, 2)                                # (B, H, S, dh)
    scores = qs @ ks.transpose(-2, -1)                    # (B, H, T, S)
    ```

    **The trap.** The inputs here are `(B, T, H, dh)`, head-minor. Half of the
    einsum's work is the implicit transpose. Writing the explicit version and
    forgetting that transpose gives `(B, T, H, S)` or a shape error, depending on
    whether `H == T`.

### Drill 29: LoRA shapes

A frozen `nn.Linear(d_in, d_out)` gets a LoRA adapter of rank `r`. Give the shapes
of `A` and `B`, the forward pass, and the parameter count against full fine-tuning.

??? success "Answer"
    ```python
    # W is (d_out, d_in) in torch's Linear convention
    A = nn.Parameter(torch.randn(r, d_in) * 0.01)         # (r, d_in)
    B = nn.Parameter(torch.zeros(d_out, r))               # (d_out, r)

    y = F.linear(x, W)                                    # (..., d_out)
    y = y + (x @ A.T) @ B.T * (alpha / r)                 # (..., r) then (..., d_out)
    ```

    Parameters: `r * (d_in + d_out)` against `d_in * d_out`. At d = 4096 and
    r = 8 that is 65k against 16.8M, a factor of 256.

    **The trap.** `B` starts at zero so the adapter is the identity at step 0 and
    the model is unchanged. Initialising both at random makes the first forward
    pass a different model from the one you fine-tuned from, and the loss spikes.
    The `alpha / r` scale is what keeps the effective learning rate comparable
    across ranks. Canon #38.

### Drill 30: group advantages

GRPO samples `G` responses for each of `P` prompts. Rewards arrive as a flat list
of length `P * G` in prompt-major order. Produce the flat advantages.

??? success "Answer"
    ```python
    r = rewards.reshape(P, G)                             # (P, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, unbiased=False, keepdim=True) + 1e-6)
    adv = adv.reshape(-1)                                 # (P*G,)
    ```

    `keepdim=True` on both statistics is what makes them broadcast back over `G`.
    The epsilon goes outside the denominator here, not inside a square root,
    because a group with identical rewards has a zero numerator too, and
    `0 / (0 + eps)` is the zero advantage you want.

    **The trap.** The flattening order. The batch must have been built with
    `repeat_interleave` (prompt-major), not `repeat` (group-major), or
    `reshape(P, G)` groups one sample from each of `G` different prompts and the
    baseline subtracts the wrong mean. Every shape is valid. Canon #56.

### Drill 31: DETR cost matrix

`pred_logits` is `(Q, K+1)` and `pred_boxes` is `(Q, 4)` for `Q = 100` queries.
There are `N_gt` ground-truth boxes. Give the shape of the matching cost and what
goes into it.

??? success "Answer"
    ```text
    class term   -pred_prob[:, gt_labels]      (Q, N_gt)
    L1 term      cdist(pred_boxes, gt_boxes)   (Q, N_gt)
    GIoU term    -generalized_iou(pred, gt)    (Q, N_gt)
    cost         weighted sum                  (Q, N_gt)
    ```

    `linear_sum_assignment(cost)` returns two index arrays of length
    `min(Q, N_gt) = N_gt`, pairing query indices with ground-truth indices. The
    other `Q - N_gt` queries are supervised toward the "no object" class.

    **The trap.** The cost matrix is rectangular and `scipy`'s solver returns the
    row and column indices as a pair of arrays, not a permutation. Treating the
    second return value as "the ground truth for query i" instead of "the ground
    truth for query `row[i]`" misassigns everything whenever `Q > N_gt`, which is
    always. Canon #40.

### Drill 32: convolution output size

Give the output spatial size for: `k=3, s=1, p=1`; `k=3, s=2, p=1`; `k=1, s=1, p=0`;
`k=7, s=2, p=3`; and a transposed convolution with `k=4, s=2, p=1`.

??? success "Answer"
    Forward: $H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$.

    ```text
    k=3, s=1, p=1   H          "same"
    k=3, s=2, p=1   ceil(H/2)  the standard downsample
    k=1, s=1, p=0   H          the channel mixer
    k=7, s=2, p=3   ceil(H/2)  the ResNet stem
    ```

    Transposed: $H_{out} = (H - 1)s - 2p + k$, so `k=4, s=2, p=1` gives exactly
    `2H`, which is why that triple is the standard upsampling block in every U-Net
    and DCGAN.

    **The trap.** `k=3, s=2, p=1` gives `ceil(H/2)` and not `H/2` when H is odd,
    so a network that downsamples five times turns 224 into 7 and 225 into 8. Skip
    connections between an encoder and a decoder then mismatch by one pixel, which
    is the single most common U-Net bug. Canon #16 and #17.

## The shape-debugging checklist

When the shapes have gone wrong and the clock is running, work down this list
rather than staring at the traceback.

1. **Print every shape.** `print(x.shape)` after each line, on `B=2, T=3, d=4`. Two minutes of this beats twenty minutes of reading.
2. **Use distinct dimension sizes.** `B=2, T=3, S=5, H=2, dh=4` makes every axis identifiable. With `B=T=d=8` a transposed tensor is indistinguishable from a correct one and every test passes.
3. **Check the last axis of the reduction.** Softmax over keys, not queries. Sum over features, not batch. Name the axis in the comment.
4. **Check `keepdims` on both operands of a broadcast.** A `(B, 1)` where you expected `(B,)` produces an outer product one line later.
5. **Look for `view` after `transpose`.** If it raised, that is the bug. If you replaced it with `reshape` and the numbers are wrong, you probably need a transpose you deleted.
6. **Test the invariant, not the output.** Softmax rows sum to one. A causal model's logit at position 3 is unchanged when you edit token 4. IoU of a box with itself is 1. A permutation of the input permutes the output. Each is one line and localises the bug to a range of lines.
7. **Check the scale.** `sqrt(dh)`, not `sqrt(d)`. Divide by the mask sum, not by `T`. `alpha / r`, not `alpha`.
8. **Check the mask value and the padding side.** Finite, not `-inf`. Right padding plus `h[:, -1]` reads a pad.
9. **Run the reference.** `torch.nn.functional.scaled_dot_product_attention`, `torchvision.ops.nms`, `torch.nn.LayerNorm`. Comparing against the built-in on a 3-token input tells you whether the bug is in the maths or in the plumbing.
10. **Say the shape out loud before typing the line.** The failure mode this prevents is writing a line whose output shape you never decided on.

## Retype by hand

Four, from a blank file, about 35 minutes total.

- [ ] Drill 6, multi-head attention with a padding mask, checked against `torch.nn.functional.scaled_dot_product_attention`.
- [ ] Drill 22 and 23, image to patches and back, checked by asserting the round trip is exact.
- [ ] Drill 25, pairwise IoU, checked against `torchvision.ops.box_iou`.
- [ ] Drill 30, group advantages, checked by asserting every row of the result has zero mean.

Then `pytest tests/test_transformer_multihead.py tests/test_detection_boxes.py tests/test_multimodal_vit.py`.

## References

* NumPy, "Broadcasting", NumPy user guide. The three-rule statement is the one to memorise.
* PyTorch, "Broadcasting semantics" and "torch.gather", official documentation.
* Alexander Rush, "Tensor Considered Harmful" (2019), on named tensors and why shape bugs survive testing.
* Tim Rocktäschel, "Einsum is All You Need" (2018), for the einsum notation used in drill 28.
