# Coding drills for the offboard perception loop

A coding round with a perception engineer is a conversation with a compiler attached. The
code matters, and so does whether you asked the right question before writing it. Every drill
here has three parts: the exchange you should have first, the implementation with the comments
you would actually write, and the tests you would write beside it unprompted.

## The protocol

Five moves, in order, every time.

1. **Restate the problem** in one sentence, including the input and output types and shapes.
2. **Ask the two or three questions that change the code.** Not trivia. Things like: can the
   input be empty, are ties possible, is the grid 4- or 8-connected, does the caller need the
   mask or the statistics, is this on the hot path.
3. **State the approach and the complexity before typing.** "Flood fill from each unvisited
   foreground pixel, $O(HW)$ time and $O(HW)$ worst-case space, iterative because recursion
   blows the stack on a real raster."
4. **Write the simple correct version.** Name things well. Comment the non-obvious line only.
5. **Write the tests, then optimise if asked.** Volunteering the tests is a large part of the
   signal at senior level, and it is free.

The one habit to practise tonight: say the complexity out loud before you type. It is the
cheapest way to sound like someone who has done this before.

---

## Drill 1: IoU and non-maximum suppression

!!! example "The exchange"
    **Interviewer.** Implement IoU for bounding boxes, then NMS.

    **You.** Boxes as `(x1, y1, x2, y2)` corners, or centre-width-height? And are they
    guaranteed valid, meaning `x2 >= x1`?

    **Interviewer.** Corners, and assume valid.

    **You.** Two more. Single class or multi-class? And for NMS, do you want the surviving
    indices or the surviving boxes?

    **Interviewer.** Single class, return indices.

    **You.** Then I will write a vectorised IoU that takes `(N, 4)` and `(M, 4)` and returns
    an `(N, M)` matrix, since NMS needs one box against many and a matrix version is more
    generally useful. NMS is sort by score descending, then repeatedly take the top box and
    drop everything overlapping it above threshold. That is $O(N^2)$ worst case and
    $O(N \log N)$ for the sort, which dominates when suppression is aggressive.

```python
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a (N, 4), b (M, 4) as (x1, y1, x2, y2) -> (N, M)."""
    # broadcast to (N, M) by inserting an axis: a[:, None] is (N, 1, 4), b[None] is (1, M, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])      # (N, M) left edge of the intersection
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])      # (N, M)
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])      # (N, M) right edge
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])      # (N, M)
    # clip at zero: a negative width means the boxes do not overlap at all
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)   # (N, M)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])              # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])              # (M,)
    union = area_a[:, None] + area_b[None, :] - inter               # (N, M)
    return inter / np.maximum(union, 1e-12)                         # guard the empty-box case


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Greedy non-maximum suppression. (N, 4), (N,) -> indices of survivors, best first."""
    order = np.argsort(-scores)          # (N,) descending score; ties broken by index
    keep = []
    while len(order):
        i, order = order[0], order[1:]   # the highest-scoring box left always survives
        keep.append(i)
        if not len(order):
            break
        ious = box_iou(boxes[i][None], boxes[order])[0]   # (len(order),)
        order = order[ious <= iou_threshold]              # drop everything it suppresses
    return np.array(keep, dtype=int)
```

**Tests you write without being asked.**

```python
def test_iou_identical_disjoint_and_contained():
    a = np.array([[0., 0., 2., 2.]])
    assert box_iou(a, a)[0, 0] == 1.0                       # identical
    assert box_iou(a, np.array([[5., 5., 6., 6.]]))[0, 0] == 0.0    # disjoint
    # half-overlap: intersection 2, union 6, so 1/3
    assert np.isclose(box_iou(a, np.array([[1., 0., 3., 2.]]))[0, 0], 1 / 3)

def test_nms_keeps_the_best_of_a_cluster_and_all_isolated_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]                          # box 1 suppressed by box 0
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.99).tolist() == [0, 1, 2]  # nothing merges

def test_nms_handles_empty_input():
    assert len(nms(np.zeros((0, 4)), np.zeros(0))) == 0
```

**The follow-up to be ready for.** "Make it faster." Answers: sort once and use a
suppression mask instead of re-slicing; batch by class with a class offset trick so one call
handles all classes; use soft-NMS (decay scores instead of deleting) when objects genuinely
overlap; or move to a detector that does not need NMS at all, which is the DETR argument.
Reference implementation: `src/mlbook/detection/nms.py` and `src/mlbook/detection/boxes.py`.

---

## Drill 2: connected components, and the landing-region question

This is the drill to over-prepare. The Zipline-shaped version is: *given a segmentation mask,
find all contiguous safe landing regions larger than N pixels.*

!!! example "The exchange"
    **Interviewer.** You are given a binary safety mask over a property. Find all contiguous
    safe regions larger than N pixels.

    **You.** Four-connected or eight-connected? It changes the answer when regions touch
    diagonally, and for a safety mask I would default to 4-connected, because a diagonal
    pixel pair is not something a vehicle can traverse.

    **Interviewer.** Four.

    **You.** And what do you want back: the labelled image, or per-region statistics?

    **Interviewer.** Statistics. Area, centroid, bounding box.

    **You.** One more, and it changes the design. Is "larger than N pixels" really the
    criterion you want? A drone needs a *disk* to fit, and an L-shaped strip can have large
    area with nowhere to land. I will implement what you asked and then show you the
    clearance version, since that is the one I would ship.

    **Interviewer.** Do both.

    **You.** Flood fill from every unvisited foreground pixel, $O(HW)$ time and $O(HW)$
    worst-case auxiliary space. Iterative rather than recursive, since a 4000 by 4000 raster
    would blow the Python stack.

```python
def connected_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Label connected regions of True. (H, W) bool -> labels (H, W) int32, count.

    Breadth-first from every unvisited foreground pixel. Each pixel enters the queue once,
    so the cost is O(HW) time and O(HW) worst-case space (a spiral fills the queue).
    """
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)              # (H, W) 0 = background
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    n = 0
    for sy in range(H):
        for sx in range(W):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            n += 1
            labels[sy, sx] = n
            queue = [(sy, sx)]
            while queue:
                y, x = queue.pop()
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    # label on ENQUEUE: labelling on dequeue lets a pixel enter many times
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = n
                        queue.append((ny, nx))
    return labels, n
```

Per-region statistics in one pass, which is the part people write as a loop over labels:

```python
def region_properties(labels: np.ndarray, n: int, gsd: float) -> list[dict]:
    """Area, centroid and bounding box for every label. O(HW + n)."""
    flat = labels.ravel()                                        # (H*W,)
    ys, xs = np.divmod(np.arange(labels.size), labels.shape[1])  # (H*W,), (H*W,)
    counts = np.bincount(flat, minlength=n + 1)[1:]              # (n,) pixels per label
    sum_y = np.bincount(flat, weights=ys, minlength=n + 1)[1:]   # (n,)
    sum_x = np.bincount(flat, weights=xs, minlength=n + 1)[1:]   # (n,)
    ...
```

`np.bincount` with `weights` computes all three accumulations in three passes over the array
instead of `n` passes over the image. At 500 regions on a 4000 by 4000 raster that is the
difference between a second and several minutes.

**Tests.**

```python
def test_connected_components_respects_connectivity():
    mask = np.array([[1,1,0,0,0],
                     [1,1,0,0,0],
                     [0,0,1,0,0],      # touches the block above only diagonally
                     [0,0,0,1,1],
                     [0,0,0,1,1]], bool)
    assert connected_components(mask, connectivity=4)[1] == 3
    assert connected_components(mask, connectivity=8)[1] == 1

def test_connected_components_on_edge_cases():
    assert connected_components(np.zeros((4, 4), bool))[1] == 0
    labels, n = connected_components(np.ones((4, 4), bool))
    assert n == 1 and (labels == 1).all()
    # a spiral is the worst case for the queue and must still terminate
    assert connected_components(spiral)[1] == 1
```

The connectivity test is the one that proves you understood the question. The empty, full and
spiral cases are the ones that prove you have written this before.

**The follow-up.** "How would you do it without a queue?" Two-pass union-find: scan
assigning provisional labels and recording equivalences in a disjoint-set structure, then scan
again resolving each to its root. Nearly linear with path compression, better cache behaviour,
and it parallelises by rows, which is why production implementations use it. Say that, then
say that for an interview you would write the flood fill because it is harder to get wrong.

Reference implementation: `src/mlbook/perception/landing_zone.py`.

---

## Drill 3: the exact distance transform and the largest inscribed disk

The clearance version of drill 2, and the one that shows judgement.

!!! example "The exchange"
    **You.** For "where can a drone actually land", the statistic is the radius of the largest
    disk that fits inside the region. That is exactly the maximum of the Euclidean distance
    transform, and its location is the argmax. Do you want an exact transform or is a chamfer
    approximation acceptable?

    **Interviewer.** What is the difference?

    **You.** Chamfer propagates integer approximations to the Euclidean metric in two raster
    passes; it is simple and a few percent off, and that error lands directly in a
    clearance measured in metres and compared against a threshold. The exact version is the
    Felzenszwalb and Huttenlocher separable transform, also $O(HW)$, which computes the lower
    envelope of a family of parabolas in one sweep per axis. I will write the exact one.

```python
def _edt_1d(f: np.ndarray) -> np.ndarray:
    """Lower envelope: D(x) = min_y (x - y)^2 + f(y). (n,) -> (n,), O(n).

    The parabolas (x - y)^2 + f(y) have a lower envelope with at most n pieces, and one
    left-to-right sweep maintaining the current envelope finds them all.
    """
    n = len(f)
    v = np.zeros(n, np.int64)          # (n,) indices of the parabolas in the envelope
    z = np.empty(n + 1)                # (n+1,) breakpoints between them
    z[0], z[1], k = -np.inf, np.inf, 0
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:               # this parabola hides the last one; pop it
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k], z[k], z[k + 1] = q, s, np.inf
    d = np.empty(n)
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def distance_transform_edt(mask: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from each True pixel to the nearest False. (H, W) -> (H, W)."""
    f = np.where(mask, 1e12, 0.0)              # (H, W) seed: cost 0 at background
    d = np.apply_along_axis(_edt_1d, 0, f)     # (H, W) squared distance down the columns
    d = np.apply_along_axis(_edt_1d, 1, d)     # (H, W) then across the rows
    return np.sqrt(d)


def largest_inscribed_disk(mask: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Radius (pixels) and centre of the largest disk fitting inside mask."""
    if not mask.any():
        return 0.0, (0, 0)
    dist = distance_transform_edt(mask)
    row, col = np.unravel_index(int(np.argmax(dist)), mask.shape)
    return float(dist[row, col]), (int(row), int(col))
```

**Tests, including the one that carries the design argument.**

```python
def test_distance_transform_matches_brute_force():
    mask = rng.random((18, 22)) > 0.25
    assert np.allclose(distance_transform_edt(mask), brute_force_edt(mask), atol=1e-9)
    # a 9x9 block of free space inside a border: the inradius is exactly 5 pixels
    block = np.zeros((11, 11), bool); block[1:10, 1:10] = True
    assert np.isclose(distance_transform_edt(block).max(), 5.0)

def test_clearance_beats_area_on_an_l_shaped_region():
    """A long thin strip has plenty of area and nowhere to land."""
    assert strip.sum() > square.sum()                 # the strip has MORE area
    assert largest_inscribed_disk(strip)[0] < largest_inscribed_disk(square)[0]
```

Checking against an $O(n^2)$ brute force is the right way to test a clever $O(n)$ algorithm,
and it is a good habit to name in the room.

---

## Drill 4: associating detections across frames by IoU

!!! example "The exchange"
    **Interviewer.** Given detections in two consecutive frames, associate them.

    **You.** Is this a one-to-one assignment, or can a detection match several? And do you
    want optimal assignment or is greedy acceptable?

    **Interviewer.** One-to-one. Tell me the difference.

    **You.** Greedy takes the highest-IoU pair, removes both, repeats. It is $O(N M \log NM)$
    and can be arbitrarily worse than optimal, because an early locally-good match can block
    two better ones. The Hungarian algorithm minimises total cost in $O(n^3)$ and is what a
    real tracker uses. I will write greedy first because it is five lines and correct enough
    to discuss, then say where I would swap in Hungarian.

    **You.** One more thing: I want a gate. A pair with IoU below a threshold should stay
    unmatched rather than being forced into an assignment. Without that, a new object entering
    the frame gets matched to an object that left.

```python
def associate_by_iou(prev_boxes, curr_boxes, iou_threshold=0.3):
    """Greedy one-to-one association. -> matches [(i, j)], unmatched_prev, unmatched_curr."""
    iou = box_iou(prev_boxes, curr_boxes)                  # (N, M)
    matches, used_prev, used_curr = [], set(), set()
    # consider candidate pairs in descending IoU
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]  # (N*M, 2)
    for i, j in order:
        if iou[i, j] < iou_threshold:     # gated: below this, leave both unmatched
            break
        if i in used_prev or j in used_curr:
            continue
        matches.append((int(i), int(j)))
        used_prev.add(i)
        used_curr.add(j)
    unmatched_prev = [i for i in range(len(prev_boxes)) if i not in used_prev]
    unmatched_curr = [j for j in range(len(curr_boxes)) if j not in used_curr]
    return matches, unmatched_prev, unmatched_curr
```

**Tests.**

```python
def test_association_is_one_to_one_and_gated():
    matches, un_p, un_c = associate_by_iou(prev, curr, 0.3)
    assert len({i for i, _ in matches}) == len(matches)     # each prev used once
    assert len({j for _, j in matches}) == len(matches)     # each curr used once
    assert all(box_iou(prev[[i]], curr[[j]])[0, 0] >= 0.3 for i, j in matches)

def test_a_new_object_stays_unmatched():
    # curr contains one box far from anything in prev
    _, _, unmatched_curr = associate_by_iou(prev, curr_with_newcomer, 0.3)
    assert newcomer_index in unmatched_curr

def test_greedy_can_be_beaten_by_optimal():
    """The failure mode worth knowing: one locally-best match blocks two better ones."""
    greedy_cost = total_iou(associate_by_iou(prev, curr)[0])
    optimal_cost = total_iou(hungarian_associate(prev, curr))
    assert optimal_cost >= greedy_cost
```

That last test is the one that shows you understand why the Hungarian algorithm exists.
Reference implementations: `src/mlbook/perception/hungarian.py` and
`src/mlbook/perception/sort_tracker.py`.

---

## Drill 5: shortest collision-free path through an occupancy grid

!!! example "The exchange"
    **Interviewer.** Find a collision-free path through an occupancy grid.

    **You.** Three questions. Is the cost uniform, or does terrain have a cost? Four-connected
    or eight-connected? And do you need the optimal path or a good one quickly?

    **Interviewer.** Uniform cost, four-connected, optimal.

    **You.** Then it is breadth-first search, $O(HW)$, and it is optimal because every edge
    costs the same so hop order is cost order. If you had told me the cost varies I would use
    Dijkstra with a heap, and if you had given me a good distance estimate I would use A*
    with an admissible heuristic, which is Dijkstra with a non-zero heuristic.

    One detail I will mention while writing: I mark cells as visited when I *enqueue* them,
    not when I dequeue. Marking on dequeue lets the same cell enter the queue many times and
    turns a linear algorithm into something much worse on open grids.

The implementation and the full test suite are in
`src/mlbook/perception/grid_planning.py` and `tests/test_perception_grid_planning.py`. The
three tests to be able to name:

```python
def test_bfs_is_optimal_on_an_open_grid():
    assert bfs_shortest_path(free, (0, 0), (7, 5), connectivity=4).cost == 7 + 5

def test_astar_matches_dijkstra_cost_and_expands_fewer_nodes():
    assert math.isclose(dijkstra.cost, astar.cost)      # admissible heuristic => same optimum
    assert astar.expanded < dijkstra.expanded           # and it gets there sooner

def test_octile_is_admissible_where_manhattan_is_not():
    # true 8-connected cost from (0,0) to (5,5) is 5*sqrt(2) = 7.07
    assert octile_distance((0, 0), (5, 5)) == 5 * math.sqrt(2)
    assert manhattan_distance((0, 0), (5, 5)) == 10     # overestimates: inadmissible
```

The heuristic-admissibility test is the differentiator. Manhattan distance with
8-connectivity is a bug that produces plausible paths and silently loses optimality, and
being the person who tests for it is a good look.

**The extension you should expect.** "Now the grid has a height map and the vehicle cannot
cross slopes above 15 degrees or steps above 20 cm." That is `traversable_from_heightmap`:
compute slope from the gradient, compute step as the maximum height difference to a
neighbour, and mask on both. Two separate constraints, because slope is a smoothed local
gradient that steps right over a kerb.

---

## Drill 6: bilinear sampling

The kernel underneath warping, `grid_sample`, RoIAlign, spatial transformers and every
resampling step in a geometry pipeline.

!!! example "The exchange"
    **You.** Two conventions to fix first. Are the coordinates pixel-centre or pixel-corner,
    meaning does integer 0 refer to the centre of the first pixel or to its top-left edge? And
    what happens outside the image: clamp, zero, or reflect?

    **Interviewer.** Pixel centre, clamp.

    **You.** Then I will clamp the integer neighbours and keep the fractional weights
    unclamped, which gives edge replication. $O(1)$ per sample, four reads and three lerps.

```python
def bilinear_sample(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sample img at fractional (y, x). img (H, W), ys/xs (N,) -> (N,)."""
    H, W = img.shape
    y0 = np.floor(ys).astype(int)                 # (N,) top neighbour row
    x0 = np.floor(xs).astype(int)                 # (N,) left neighbour column
    wy, wx = ys - y0, xs - x0                     # (N,) fractional parts, in [0, 1)
    # clamp the indices, not the weights: that is what makes out-of-bounds replicate the edge
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
    top = img[y0c, x0c] * (1 - wx) + img[y0c, x1c] * wx        # (N,) lerp along x, top row
    bot = img[y1c, x0c] * (1 - wx) + img[y1c, x1c] * wx        # (N,) lerp along x, bottom row
    return top * (1 - wy) + bot * wy                            # (N,) lerp along y
```

**Tests.**

```python
def test_bilinear_is_exact_at_integer_coordinates():
    assert np.allclose(bilinear_sample(img, np.array([2.0]), np.array([3.0])), img[2, 3])

def test_bilinear_midpoint_is_the_mean_of_four_neighbours():
    got = bilinear_sample(img, np.array([1.5]), np.array([1.5]))
    assert np.isclose(got, img[1:3, 1:3].mean())

def test_bilinear_reproduces_a_linear_ramp_exactly():
    """Bilinear interpolation is exact for functions linear in x and y."""
    ramp = 3.0 * np.arange(8)[None, :] + 2.0 * np.arange(8)[:, None]
    assert np.allclose(bilinear_sample(ramp, np.array([2.4]), np.array([5.7])),
                       2.0 * 2.4 + 3.0 * 5.7)
```

The ramp test is the good one: it checks the mathematical property rather than one hand-computed
number, so it catches a swapped `wx`/`wy` that the midpoint test passes by symmetry.

Reference implementation: `src/mlbook/vision/image_ops.py`.

---

## Drill 7: tensor fluency, five minutes each

Short ones that show up as warm-ups or as a follow-up to something larger. Each should take
under five minutes.

**Softmax attention.** One head, explicit, no fused projections.

```python
def attention(Q, K, V, mask=None):
    """Q (B, T, d), K (B, S, d), V (B, S, d) -> (B, T, d)."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])   # (B, T, S)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))   # (B, T, S)
    weights = torch.softmax(scores, dim=-1)                     # (B, T, S) rows sum to 1
    return weights @ V                                          # (B, T, d)
```

The scaling by $\sqrt{d}$ is the thing to explain: dot products of independent unit-variance
vectors have variance $d$, so without it the softmax saturates and gradients vanish as $d$
grows.

**Cross entropy from logits**, with the stability trick:

```python
def cross_entropy(logits, targets):
    """logits (N, K), targets (N,) int -> scalar."""
    m = logits.max(dim=1, keepdim=True).values          # (N, 1) subtract the max first:
    z = logits - m                                       # (N, K) exp() cannot overflow now
    logZ = z.exp().sum(dim=1).log() + m.squeeze(1)      # (N,) log-sum-exp
    return (logZ - logits.gather(1, targets[:, None]).squeeze(1)).mean()
```

**Cosine similarity** between two sets of embeddings, with the epsilon that stops a zero
vector producing a NaN:

```python
def cosine_similarity(A, B, eps=1e-8):
    """A (N, d), B (M, d) -> (N, M)."""
    A = A / A.norm(dim=1, keepdim=True).clamp(min=eps)   # (N, d)
    B = B / B.norm(dim=1, keepdim=True).clamp(min=eps)   # (M, d)
    return A @ B.T                                       # (N, M)
```

**A training loop**, asked more often than people expect, and worth being able to type without
thinking:

```python
for epoch in range(n_epochs):
    model.train()
    for x, y in loader:
        opt.zero_grad(set_to_none=True)     # set_to_none is faster and avoids stale-grad bugs
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        ...
```

The details that signal experience: `zero_grad` before the forward pass, gradient clipping,
the scheduler stepping per batch or per epoch deliberately, and `model.eval()` with
`torch.no_grad()` around validation.

More of these in [Part XVI, the coding canon](../part16-coding-canon/index.md).

---

## What to practise tonight, in priority order

1. **Connected components**, both connectivities, plus region statistics. Twice.
2. **The distance transform and largest inscribed disk**, at least once, and be able to say
   why the clearance radius beats area even if you cannot recall the parabola sweep.
3. **IoU and NMS**, until it is muscle memory.
4. **BFS on a grid**, with the enqueue-marking detail.
5. **Bilinear sampling**, for the convention discussion as much as the code.

Everything else is a bonus. If you only get through the first two, you have covered the drill
most likely to appear in this specific loop.
