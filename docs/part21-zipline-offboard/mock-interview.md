# A 45-minute mock interview

A constructed dialogue, written so you can read the interviewer's lines aloud and answer
before reading on. The interviewer here is a composite of a robotics and computer-vision lead
with a drone background: someone who will interrupt, follow the thread you offer, and probe
for whether you have shipped something and not merely read about it. It is a rehearsal device,
and not a prediction of what any particular person will ask.

Read it twice. The first time, cover the answers. The second time, read the annotations.

!!! tip "How to use this"
    Set a 45-minute timer. Answer out loud. If you catch yourself giving a list where the
    dialogue gives a decision, that is the gap to close tonight.

---

## Minutes 0 to 4: the opening

**Interviewer.** Tell me about what you have been working on.

**You.** At Google I worked on large-scale visual perception for Maps and Street View. The
systems combined detection, classification, OCR and geometric signals, running inference over
extremely large image collections. The piece I am most proud of was moving from several
sequential models to a shared visual encoder with task-specific heads, which cut redundant
compute and let the tasks share representation.

*Then stop.*

!!! note "Annotation"
    Four sentences, one concrete architectural decision, then silence. The silence is the
    technique. You have offered three threads (scale, multi-task, geometry) and let the
    interviewer pick, which means the next ten minutes are on ground you chose. A five-minute
    monologue spends the same time and surrenders that control.

**Interviewer.** Why a shared encoder?

**You.** Two reasons, and they pulled in the same direction. Compute: the encoder dominated
cost and we were running it several times over the same pixels. And representation: the tasks
were correlated, so features learned for one helped another, particularly for the
lower-frequency classes where any single task had thin data.

**Interviewer.** When does multi-task hurt?

**You.** When the tasks want incompatible features, when the dataset sizes are very
different so one task's gradients dominate, and when the loss scales are mismatched so the
effective weighting is an accident of units instead of a decision.

**Interviewer.** So how do you fix it?

**You.** In the order I would actually try them. Loss weighting, first with a manual sweep
and then with uncertainty weighting if there are many tasks. Sampling, to stop the large
dataset dominating each batch. Architecture, by branching later so the tasks share only the
layers where sharing helps, or by adding task-specific adapters. And if gradients genuinely
conflict, gradient surgery, though I would treat that as evidence the task grouping is wrong
instead of treating it as the fix.

!!! note "Annotation"
    Note the shape: ordered by cost, with a statement about when a technique is a symptom
    instead of a solution. That last clause is the staff move.

---

## Minutes 4 to 22: the system design question

**Interviewer.** Here is what we actually do. We have aerial survey imagery of a customer's
property. Before a drone ever attempts a delivery, we need to determine whether a location is
safe and suitable. Design the perception system.

**You.** Before I design it, four questions, because each one changes the answer.

What does the vehicle consume: a binary gate, ranked candidate sites, or a dense prior it
fuses with its own perception?

**Interviewer.** A prior. It has its own downward camera on descent.

**You.** Good. Then I am not the last line of defence, which changes my objective from "be
right" to "be correctly conservative and explicit about what I do not know".

Second: what does the delivery physically require? A landing footprint, a hover and winch?

**Interviewer.** A droid descends on a tether from a few hundred feet and steers itself onto
roughly a one metre target.

**You.** Then my constraint is a vertical corridor, not a ground patch, and thin structures in
that corridor are my dominant hazard. Third: how fresh is the imagery?

**Interviewer.** Survey imagery, possibly months old.

**You.** So staleness is a first-class problem, and the prior needs a validity horizon rather
than being a static asset. Last one: the cost ratio between declining a good site and
attempting a bad one?

**Interviewer.** Wildly asymmetric.

**You.** Then the operating point is essentially determined by that ratio, and I want it to be
an explicit number with an owner outside the model.

!!! note "Annotation"
    Ninety seconds. Four questions, and each answer visibly changed the design. This is the
    single highest-leverage thing in the whole interview, because it converts a
    guess-what-I-am-thinking exercise into a conversation between two engineers.

**You.** Here is the architecture. Three layers.

Geometry, from multi-view reconstruction: a surface model, a bare-earth model, slope, local
relief, height above ground, and a volumetric occupancy representation over the corridor above
each candidate site. Every cell carries a reconstruction confidence.

Semantics, from a segmentation model over the same imagery: surface classes (grass, driveway,
patio, roof, road, water) and hazard classes (tree, wire, pole, vehicle, person, pool, fence).
Thin structures get a dedicated detector, because at survey resolution they are sub-pixel.

Then a decision layer, which is where I would spend the design effort. It is a conjunction of
hard constraints, each independently checkable and individually logged: slope under a
threshold, a clear disk of the required radius, hazard probability under a budget with a
spatial buffer, and a clear vertical corridor. Sites that survive are then ranked.

**Interviewer.** Why not just train a model to predict deliverability end to end?

**You.** Eventually I might, on top of this, once there are outcome labels from real
deliveries. Not in version one, for three reasons.

Auditability: when the system declines a property, someone has to be able to say which
constraint failed. A single learned scalar cannot answer that, and that answer is what makes
the system operable.

Data: I do not have deliverability labels yet. I have geometry, semantics and a small number
of outcomes. An end-to-end model would be fitting a very rare event from very little data.

Change management: the safety thresholds will move, because regulators, insurers and incidents
will move them. Moving a threshold should be a config change with a review, not a retraining
run.

!!! note "Annotation"
    Three reasons in three different registers: engineering, statistical, organisational. A
    candidate who gives three engineering reasons sounds senior. A candidate who reaches for
    change management sounds staff.

**Interviewer.** Tell me about the clearance constraint. How do you compute it?

**You.** The naive version is area: take connected components of the safe mask and keep those
above some pixel count. That is wrong, and it is worth saying why. A drone needs a disk. An
L-shaped strip between a fence and a shed can have twenty square metres of area and nowhere to
put a one-metre circle.

So the statistic is the largest inscribed circle, which is exactly the maximum of the
Euclidean distance transform of the safe mask, and its location is the argmax. Linear time
with the separable exact transform, and it gives me the touchdown point and the clearance
radius together.

**Interviewer.** How accurate does that radius need to be?

**You.** It needs to be accurate in metres, so it comes back to ground sample distance. At 90
metres altitude with a 2400 pixel focal length the GSD is under four centimetres, so a one
metre target is about 27 pixels. A two pixel error in the mask boundary is eight centimetres
of clearance, which is fine against a 1.5 metre requirement and not fine if the margin is
tight. Which is why I would apply the constraint to the lower confidence bound of the radius
instead of to the point estimate, and carry a per-site reconstruction uncertainty in place of
a global assumption.

**Interviewer.** Say more about the corridor.

**You.** A surface model answers the ground question and misses everything overhead: a branch
at six metres, a service drop crossing the yard at eight. Since the droid descends from
altitude, the constraint is that a cylinder of the required radius, from release height down
to the ground, contains no occupied voxel.

That makes wires my hardest problem, and I would raise it before you do. At four centimetre
GSD a typical conductor is a fraction of a pixel wide. Per-pixel segmentation on a single
frame will not find it reliably, and no amount of model scale fixes a resolution limit.

Three things that do work, and I would use all three. Exploit the shape: a wire is thin and
very long, so a detector that integrates evidence along a catenary has orders of magnitude
more signal than any single pixel. Exploit multi-view: it appears in many overlapping frames
along a predictable epipolar path, so consistency separates it from linear ground texture.
Exploit context: wires terminate at poles, poles are large and easy to detect, so detect the
endpoints and hypothesise the span.

And then the system-level answer, which matters more than the detector. I would not rely on
detection recall alone for a safety decision. Regions near any detected or inferred pole are
unsafe by default, utility infrastructure data feeds in where it exists, and anywhere I cannot
resolve the question is marked unknown instead of clear.

!!! note "Annotation"
    Raising the hardest sub-problem before being asked, quantifying why it is hard, giving
    three independent attacks, and then explicitly saying the model is not the mitigation.
    That last move is the difference between a strong ML engineer and someone you would trust
    with a safety-critical system.

**Interviewer.** How would you stage this? Say you are starting from nothing.

**You.** Five phases, each with a promotion criterion decided before the work starts.

Phase 0 is rules and classical geometry: photogrammetry to a surface model, morphological
bare-earth extraction, hand-written constraints. No learning. It is the baseline everything
else is measured against, and it builds the review tooling. I promote when the remaining
errors are semantic instead of geometric, because that tells me the next investment is a
semantic model.

Phase 1 is supervised segmentation feeding the same constraint structure. Promote on
per-class recall for hazard classes at the operating point, measured on held-out
*properties*, since tiles from one property leak.

Phase 2 is learned multi-view geometry for coverage where classical matching fails, with the
classical pipeline retained as a verifier on a routed subset. Promote on agreement within a
metric tolerance where classical succeeds, plus measured coverage gain where it does not.

Phase 3 is richer priors and open-vocabulary detection for the long tail, which only makes
sense once the evaluation infrastructure exists to see long-tail movement.

Phase 4 is the continuous loop: hard examples mined from onboard disagreement, operator
overrides and aborted deliveries. Every delivery is a free supervised test of the prior, and
that is the signal I would build the data engine around.

---

## Minutes 22 to 32: the geometry drill

**Interviewer.** Let us go down a level. You said multi-view reconstruction. Walk me through
how you get depth from two images.

**You.** Triangulation. Each pixel is a ray through its camera centre, and with known relative
pose two rays intersect at the 3-D point. Linearly, from $p \sim PX$ the cross product
$p \times PX = 0$ gives two equations per view, so two views give a four by four homogeneous
system solved by SVD. Then refine on true reprojection error, because the linear solution
minimises an algebraic quantity.

**Interviewer.** What makes it unreliable?

**You.** The dominant term is geometric. In the rectified case $Z = fB/d$, so
$\sigma_Z = Z^2\sigma_d/(fB)$. Error grows with the *square* of depth and falls linearly with
baseline.

For our numbers: 90 metres altitude, a 10 metre baseline between frames, 2400 pixel focal
length, feature localisation good to 0.3 pixels. That gives about 10 centimetres of depth
noise. Halve the baseline and it is 20 centimetres, which is already inside a 25 centimetre
relief threshold. At that point the system cannot distinguish a flat lawn from a flower bed.

Beyond geometry: pose error, calibration error, poor feature localisation, repeated texture
producing confident wrong matches, moving vegetation, water, and occlusion.

**Interviewer.** You have hundreds of frames from one flight. How do you pick pairs?

**You.** This is where the obvious thing is wrong, so I would flag it. Consecutive frames have
the most matches and the worst geometry, because they are metres apart at 90 metres altitude.
Selecting by match count systematically selects the worst-conditioned pairs.

I would select by triangulation angle subject to a minimum match count, keep sequential
neighbours for track continuity, and deliberately add cross-strip edges, because those carry
the wide baselines that actually constrain depth. And I would use the INS poses to build the
candidate graph instead of matching all pairs, which is quadratic and dies at scale.

**Interviewer.** Suppose the scene is a flat field. What happens?

**You.** Degenerate. A homography explains every correspondence, so the fundamental matrix is
not uniquely determined and RANSAC returns one member of a family with a high inlier count.
Triangulating with it gives a plausible reconstruction that is wrong.

Detection is cheap: fit $H$ alongside $F$ and compare inlier counts and residuals. If $H$
explains the matches as well, treat the pair as degenerate. The same test catches the other
degeneracy, pure rotation, which a hovering drone produces constantly.

**Interviewer.** Now suppose I tell you we are using a feed-forward model that predicts 3-D
directly. Why would you keep any of this?

**You.** Because the failure modes are different in a way that matters more than the accuracy
numbers.

Classical SfM fails loudly: no match, high residual, a hole in the output. The decision layer
treats a hole as unknown and declines, which is the safe direction. A learned model fails
quietly: it produces a confident, smooth, plausible surface where it has no evidence, and the
decision layer accepts it. You can have the same error rate and much worse outcomes.

So I would use the learned model for coverage, which is real, and keep explicit geometry as
the verifier: reprojection error against the images, agreement with triangulated depth where
both exist, agreement between predicted camera centres and GNSS baselines, and multi-view
consistency. The learned part earns coverage. The geometric part earns a residual I can
threshold and alert on.

**Interviewer.** And splats? We have looked at Gaussian splatting.

**You.** For appearance, novel-view synthesis, simulation and human review, yes. For metric
geometry feeding a safety decision, I would be cautious, and the reason is specific rather
than general.

A splat scene is optimised against a photometric loss, which is fully satisfied by any
configuration that renders correctly from the training views. Floating semi-transparent blobs
do exactly that. The expected depth from alpha compositing can sit between two surfaces and
correspond to nothing physical, which is precisely what happens near a canopy or a wire.

So high render quality is strong evidence about appearance and weak evidence about geometry. I
would derive the safety geometry from something with a validated error model, then use
disagreement between the splat reconstruction and that geometry as an uncertainty signal. A
site where they disagree beyond a threshold is a site to decline instead of guess at.

!!! note "Annotation"
    Notice the pattern across all three answers: name the failure mode, say whether it is
    loud or quiet, and then reason about what the downstream decision does with each. That
    framing is portable to almost any question in this loop.

---

## Minutes 32 to 42: the coding question

**Interviewer.** Let us write some code. I will give you a binary segmentation mask marking
safe pixels. Find all contiguous safe landing regions larger than N pixels.

**You.** Four-connected or eight-connected? It changes the answer when regions touch
diagonally, and for a safety mask I would default to four, since a diagonal pixel pair is not
something a vehicle can use.

**Interviewer.** Four.

**You.** And do you want the labelled image back, or per-region statistics?

**Interviewer.** Statistics.

**You.** One more, and it changes what I would ship. Is "larger than N pixels" the criterion
you want? A drone needs a disk to fit, and an L-shaped strip can have large area with no
usable circle. I will write what you asked, then show you the clearance version.

**Interviewer.** Go ahead.

**You.** Flood fill from every unvisited foreground pixel. $O(HW)$ time, $O(HW)$ worst-case
auxiliary space since a spiral can fill the queue. Iterative, because recursion blows the
Python stack on a real raster.

```python
def connected_components(mask, connectivity=4):
    """Label connected regions of True. (H, W) bool -> labels (H, W) int32, count."""
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)
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
                    # label on ENQUEUE, or a pixel can enter the queue many times
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = n
                        queue.append((ny, nx))
    return labels, n
```

**You.** Tests I would write beside it. Connectivity, because that was the clarifying question
and it deserves an assertion. Then the empty mask, the fully-true mask, and a spiral for the
queue worst case.

```python
def test_connectivity():
    assert connected_components(diagonal_touch, 4)[1] == 3
    assert connected_components(diagonal_touch, 8)[1] == 1

def test_edge_cases():
    assert connected_components(np.zeros((4, 4), bool))[1] == 0
    labels, n = connected_components(np.ones((4, 4), bool))
    assert n == 1 and (labels == 1).all()
```

**Interviewer.** Now the clearance version.

**You.** The radius of the largest inscribed disk is the maximum of the Euclidean distance
transform of the region, and the touchdown point is its argmax. Two lines given the transform.

```python
def largest_inscribed_disk(mask):
    """Radius in pixels and centre of the largest disk fitting inside mask."""
    if not mask.any():
        return 0.0, (0, 0)
    dist = distance_transform_edt(mask)          # distance to the nearest blocked pixel
    row, col = np.unravel_index(int(np.argmax(dist)), mask.shape)
    return float(dist[row, col]), (int(row), int(col))
```

And the test that carries the argument:

```python
def test_clearance_beats_area_on_an_l_shaped_region():
    assert strip.sum() > square.sum()                                  # MORE area
    assert largest_inscribed_disk(strip)[0] < largest_inscribed_disk(square)[0]   # LESS clearance
```

**Interviewer.** How would you implement the distance transform?

**You.** Two options and I would name the trade. Chamfer, two raster passes propagating
integer approximations, simple and a few percent off. Or the exact separable transform, which
is also linear: for each column compute the lower envelope of the parabolas
$(x-y)^2 + f(y)$ in one sweep, then do the same across rows. I would use the exact one here,
because the output becomes a clearance in metres compared against a threshold, and a few
percent of 1.5 metres is several centimetres of real margin.

**Interviewer.** And if the raster is 20,000 by 20,000?

**You.** Then the Python loop in the flood fill becomes the bottleneck before the algorithm does. Three
options in order: tile the raster and stitch labels at the boundaries, which also parallelises
and is what I would do at scale; switch to two-pass union-find, which vectorises the first
pass and has better cache behaviour; or push the inner loop into a compiled kernel. The
distance transform is already separable, so it parallelises over rows and columns directly.

!!! note "Annotation"
    The coding segment worked because of what surrounded the code: the clarifying questions,
    the stated complexity, the unprompted tests, the proactive better-design suggestion, and
    a scaling answer that names tiling before reaching for a faster language.

---

## Minutes 42 to 45: your questions

Ask two or three, and make them the questions of someone deciding whether to join.

* How does the offboard team decide what to ship to the vehicle as a prior, and who owns the
  thresholds that turn perception outputs into a go or no-go?
* What is the feedback loop like today from a delivery that went wrong back to a change in the
  offboard models? How long is that cycle?
* Where is the current bottleneck: reconstruction quality, semantic coverage on the long tail,
  or the evaluation infrastructure to know which of those is the problem?
* How are the offboard and onboard perception teams organised, and how do disagreements
  between the prior and live perception get triaged?

The third one is the best of these, because the answer tells you what the job actually is for
the first six months, and asking it signals that you think in those terms.

---

## Grading rubric

Score yourself after the run.

| Dimension | Weak | Strong | Staff |
|---|---|---|---|
| **Clarifying** | Starts designing immediately | Asks two or three good questions | Each question visibly changes the design, and says how |
| **Problem framing** | Starts from the model | Starts from the product output | Names the vehicle contract, the failure model and the cost asymmetry |
| **Geometry fluency** | Recites definitions | Derives the epipolar constraint and triangulation | Volunteers the uncertainty formula with numbers for this system |
| **Learned vs classical** | Picks a side | Lists trade-offs both ways | Refuses the dichotomy and specifies the validation that makes the hybrid safe |
| **Failure modes** | Generic (overfitting, distribution shift) | Specific to aerial (vegetation, wires, water, repeated texture) | Classifies failures as loud or quiet and reasons about what downstream does with each |
| **Evaluation** | Accuracy and mIoU | Per-class, slices, calibration | Decision-level metrics, the rule of three, and why the release gate cannot be the safety rate |
| **Coding** | Correct code | Correct, with complexity stated | Clarifies first, tests unprompted, proposes the better design, scales it |
| **Roadmap** | "I would train X" | Phased plan | Phases with promotion criteria decided in advance |
| **Ownership** | "The team did" | "I did" | What you owned, what it cost, what you would do differently |

If you are at "strong" across the board you interview well. The staff column is where the
title conversation happens, and most of it is a habit of saying the second sentence: not just
the decision, but what the decision commits you to.

---

## Six traps in this specific loop

1. **Answering the segmentation question.** "Design the perception system" invites "I would
   train a segmentation model". Start with the vehicle contract.
2. **Using area instead of clearance.** If you say "regions larger than N pixels" and never
   correct it, you have missed the one piece of domain reasoning the question was testing.
3. **Forgetting the corridor.** A ground-only answer solves the easier half of a tethered
   delivery.
4. **Defending deep learning.** "Why feed-forward instead of SfM" is not a loyalty test. The
   answer is about failure modes and validation.
5. **Quoting mIoU as the safety metric.** Have the wire arithmetic ready: 0.05% of pixels,
   99.95% accuracy, zero recall.
6. **Passive project stories.** "We built" is invisible. Say what you decided, what it cost,
   and what you would change.

---

## The five questions, once more

If you can answer these fluently and out loud, you are ready.

1. You have 20 overlapping aerial images with approximate camera poses. Build a metrically
   accurate 3-D representation. ([chapter 2](02-two-view-geometry.md) and
   [chapter 3](03-sfm-and-bundle-adjustment.md))
2. Design a system that determines whether a backyard location is safe for autonomous
   delivery. ([chapter 5](05-deliverability-system-design.md))
3. Your segmentation model has 97% accuracy but occasionally misses wires and tree branches.
   What do you do? ([chapter 6](06-evaluation-and-scaling.md))
4. Your new learned 3-D reconstruction beats SfM offline but causes more failures in
   production. How do you debug it? ([chapter 6](06-evaluation-and-scaling.md))
5. You need to scale perception from 10,000 properties to 10 million. What changes
   architecturally? ([chapter 6](06-evaluation-and-scaling.md))
