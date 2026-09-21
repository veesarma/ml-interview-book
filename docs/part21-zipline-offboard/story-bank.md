# Story bank: translating Street View perception into aerial terms

Fifteen percent of the loop, and the part most likely to decide whether the technical answers
are believed. Your Google work is closer to this role than almost any other candidate's will
be, and the job in the room is to make the mapping obvious without overclaiming.

Use [the CARL framework](../part20-project-deep-dive/01-carl-framework.md) for structure. This
page is the Zipline-specific translation and the follow-up map.

## The one-paragraph translation

The conceptual move is a short one, and saying it explicitly costs fifteen seconds and buys
the rest of the conversation.

| Street View | Zipline offboard |
|---|---|
| Street-level imagery at continental scale | Aerial survey imagery per property |
| Detection, classification, OCR, geometric signals | Segmentation, reconstruction, hazard detection |
| Semantic understanding of a scene | Semantic **and metric** understanding of a site |
| A map representation | A 3-D world representation plus a deliverability prior |
| Downstream Maps products | Downstream autonomous delivery |
| Wrong label degrades a product surface | Wrong prior contributes to a physical incident |

The two rows that carry the weight are the last two. The addition is **metric geometry with an
error bar**, and the change in consequence is from product quality to safety. Name both as the
things you are stepping into rather than things you already have, which is more credible and
happens to be true.

!!! tip "The framing sentence"
    "My production perception experience transfers directly, and the two things I would be
    learning are metric 3-D geometry as a first-class output and the discipline of a
    safety-critical decision layer. That second one is why I have been reading how Waymo
    structures a safety case."

    That sentence does more for you than any claim of expertise, because it tells the
    interviewer you know what the gap is, which is what they are trying to work out.

## The opener, and the silence

Four sentences, then stop.

> At Google I worked on large-scale visual perception for Maps and Street View. The systems
> combined detection, classification, OCR and geometric signals, with inference running over
> extremely large image collections. The evolution I am most proud of was moving from several
> sequential models toward a shared visual encoder with specialised heads, which cut redundant
> compute while letting the tasks share representation.

Three threads offered: scale, multi-task architecture, geometry. Whichever the interviewer
pulls, you are on prepared ground.

The temptation is to keep talking. Do not. A four-sentence opener followed by silence reads as
confidence and hands the interviewer control of the next question, which is the outcome you
want.

## The follow-up map

Every one of these has been asked of someone giving that opener.

!!! interview "Why a shared encoder?"
    Compute and representation, pulling the same way. The encoder dominated cost and was being
    run repeatedly over the same pixels. And the tasks were correlated, so shared features
    helped, particularly on lower-frequency classes where any single task had thin data.

    Have a number ready if you have one: the compute reduction, the latency change, the
    throughput at fixed cost. If you do not have a number you can share, say what it was
    measured in and that you cannot share the figure, which is better than a vague adjective.

!!! interview "When does multi-task learning hurt?"
    When tasks want incompatible features, so capacity is spent reconciling them. When dataset
    sizes differ enough that one task's gradients dominate each batch. When loss scales are
    mismatched, so the effective weighting is an accident of units. And when the tasks have
    different optimal training schedules, so one is undertrained when the other converges.

!!! interview "How would you fix it?"
    In cost order. Loss weighting, manual first, then uncertainty weighting if there are many
    tasks. Sampling, to stop the large dataset dominating. Architecture, branching later or
    adding task-specific adapters so sharing happens only where it helps. Gradient surgery
    last, and I would treat needing it as evidence the task grouping is wrong.

!!! interview "How did you know the new system was better?"
    Not overall accuracy, which was the least informative number available.

    Per-class precision and recall, because the aggregate was dominated by the head classes.
    Tail performance specifically, since that was the point of sharing representation.
    Calibration, since downstream consumers thresholded the scores. Failure slices, by region,
    capture condition and imagery age. Then the systems metrics: latency, compute per image,
    throughput at fixed cost. Then regression analysis against the previous system, case by
    case on the disagreements, because an aggregate improvement can hide a class of new
    failures. And finally the production impact on the downstream surface, which is the only
    number anyone outside the team cared about.

    The habit to convey: a change ships when you can say what got worse, not only what got
    better. Every real change makes something worse.

!!! interview "What would you do differently?"
    Have a real answer. The strongest version names something structural rather than a
    tactical regret: an evaluation set you would have built earlier, an interface you would
    have versioned from the start, a decision you made for delivery speed that cost more later
    than it saved. Then say what it cost, in time or in incidents, and what you changed in how
    you work.

## Four stories to have loaded

Prepare these as CARL, not as narration. Two minutes each, with a decision at the centre.

**The architecture change.** The shared encoder. Centre it on the decision and the risk: what
the alternative was, why you chose shared over separate, what you measured to be sure, and
what you gave up. The give-up is the interesting part, since a shared encoder couples the
release cycles of tasks that used to be independent.

**The scale story.** Something about operating perception over very large image volumes: the
cost model, what you sharded on, what you cached, where the bottleneck actually was against
where you expected it. This maps directly onto the 10,000 to 10 million question, so rehearse
it with numbers.

**The failure story.** A production failure you owned. What broke, how you found out (and how
long that took, honestly), what the fix was, and what you changed so the class of failure
could not recur silently. For a safety-critical interviewer this is the most informative story
you can tell, and a candidate with no failure story reads as either junior or evasive.

**The disagreement story.** A technical disagreement where you were wrong, or where you were
right and had to bring someone along. Staff roles are partly about this and interviewers probe
for it. Keep it specific and keep the other person's position fairly stated, because a story
where the other party is an idiot tells the interviewer something about you.

## Connecting each story to this role

After every story, one sentence landing it in their world. Not laboured, just placed.

* After the architecture story: "The same question shows up here as whether reconstruction and
  semantics share a backbone, and I would expect the answer to depend on whether the geometry
  head needs different features than the semantic head."
* After the scale story: "Which is the shape of going from thousands of properties to
  millions, where I would expect cost per property to become the design constraint."
* After the failure story: "And the reason I keep coming back to explicit unknowns and
  deterministic replay is that this failure would have been found in a day with them."

## What to ask about your gap, before they find it

If metric geometry is your thinnest area, do not wait to be caught. Mid-conversation, after
you have earned some credibility:

> The part of this I would be ramping on is metric 3-D geometry as a shipped product rather
> than as a signal. I have the fundamentals, and what I would want to learn quickly is how
> your reconstruction error budget actually decomposes in practice and where it binds first.

Two effects. It is honest, which the technical answers have already made credible. And it is a
good question, which turns a gap into a conversation about the work.

## The questions to ask at the end

Pick two or three. The third is the strongest.

* How does the offboard team decide what to ship to the vehicle as a prior, and who owns the
  thresholds that turn perception outputs into a go or no-go?
* What does the feedback loop look like today from a delivery that went wrong back to a change
  in the offboard models, and how long is that cycle?
* Where is the current bottleneck: reconstruction quality, semantic coverage on the long tail,
  or the evaluation infrastructure to know which of those is the problem?
* How are offboard and onboard perception organised, and how do disagreements between the
  prior and live perception get triaged?
* For a staff-level scope here, would you expect someone to own a model roadmap end to end, or
  to work across several offboard models?

The last one is worth asking if the level is still open, because it asks about scope without
asking about title.

## Related reading

* [The CARL framework](../part20-project-deep-dive/01-carl-framework.md) for the structure.
* [Behavioral and leadership](../part20-project-deep-dive/04-behavioral-leadership.md) for the
  non-technical half.
* [Worked deep dive, StreetSmart](../part20-project-deep-dive/02-streetsmart-deep-dive.md) for
  a full example at length.
