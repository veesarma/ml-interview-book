# Behavioral and leadership rounds for staff ML engineers

> **Why this matters at staff level.** The behavioral round decides level more often than the
> technical rounds do. Two candidates can be equally strong on attention mathematics; what
> separates a senior offer from a staff offer is evidence that you owned outcomes, changed
> what other people did, were wrong in public, and recovered. This chapter is a question
> bank with the grading criteria attached, mapped onto the two projects you actually have.

## TL;DR: the interview card

* Eight prepared stories cover about thirty questions. Build the inventory, not a script per
  question.
* Every story needs four parts: the decision you owned, the alternative you rejected, the
  number that moved, and what it cost.
* Spend 60% of each answer on what *you* did. "We" is the most common reason a staff
  candidate gets levelled down.
* Two to three minutes per answer. Then stop and let them follow up. Long answers read as
  unfiltered.
* Your strongest story is the StreetSmart regression you caused, diagnosed and bounded. Lead
  with it when asked about failure, and volunteer it when asked about shipping judgment.
* ML-specific questions generic guides skip: readiness to ship, killing your own project, the
  metric disagreeing with the product, labelling-quality disputes, setting an evaluation bar
  against a partner team. These are where staff signal concentrates.
* Amazon runs this explicitly against 16 published Leadership Principles with a Bar Raiser in
  the loop ([Amazon: Leadership
  Principles](https://amazon.jobs/content/en/our-workplace/leadership-principles);
  [the interview loop](https://amazon.jobs/content/en/how-we-hire/interview-loop)). Map your
  stories to them in advance.
* Rehearse the follow-up, not the story. Interviewers grade what happens after "tell me more
  about your part in that".

## 1. What the interviewer writes down

A behavioral interviewer is collecting evidence against a small number of dimensions, and
almost everything they ask is a route into one of them.

| Dimension | The question behind the question | What counts as evidence |
|---|---|---|
| Scope of ownership | how big a thing have you been responsible for, end to end | a decision that affected other teams, a system you were paged for, a rollout you owned |
| Judgment under uncertainty | do you act when the data is incomplete, and do you know what you were betting | a decision made with a named unknown, and the experiment you ran to shrink it |
| Influence without authority | can you change what other people build | a proposal someone else implemented, a standard adopted, a review that changed a design |
| Self-awareness | can you describe your own failure precisely | a failure you caused, with mechanism, bound, and what you changed afterwards |
| Communication and conflict | what happens when someone disagrees with you | a disagreement with a resolution, including the cost of the resolution |
| Raising others | do people around you get better | a specific person, a specific change in what they could do |

The sentence shape that produces evidence in every one of them: *we had this constraint, I
chose this over that because of this mechanism, it moved this number by this amount, and it
cost us this.* If your story cannot be compressed into that shape, it is an anecdote.

## 2. Build the inventory, not the scripts

Eight stories, each usable for several questions. Fill this table before you prepare any
individual answer.

| # | Story | Source | Covers |
|---|---|---|---|
| 1 | The OCR-reliability regression you caused and bounded | StreetSmart | failure, production incident, ownership, self-awareness, shipping judgment |
| 2 | The decision rule that disqualified the backbone fine-tune | StreetSmart | decision under incomplete information, saying no, judgment, influence |
| 3 | The EU non-regression bound held against pressure to ship | StreetSmart | customer focus, insisting on a standard, disagreement |
| 4 | Getting OCR reuse accepted as a feature dependency | StreetSmart | influence without authority, working across teams |
| 5 | The 6-week backfill window and what you descoped to hold it | StreetSmart | prioritising under a deadline, delivering results |
| 6 | The quantization and latency work under a hard budget | Epirus | constraints, bias for action, technical depth on hardware |
| 7 | **[FILL: a mentoring or hiring story. A specific person, what they could not do before, what they could do after, and what you did.]** | either | mentoring, raising others, hire and develop |
| 8 | **[FILL: a disagreement with your own manager or a senior stakeholder, with its resolution and its cost.]** | either | backbone, disagree and commit, conflict |

Stories 7 and 8 are the ones that do not appear in a project deck, and they are asked in
almost every staff loop. Write them now rather than discovering the gap in the room.

## 3. The standard question bank

For each: what is being graded, where your material is, and the structure of an answer.
Rehearse the structure, not the words. A memorised answer is audible and it does not survive
"tell me more about that part".

### "Tell me about a conflict with a partner team."

*Graded on:* whether you can disagree without escalating, and whether you know what the other
team's incentive actually was.

*Structure.* Name the disagreement as a legitimate difference of interests before you name
your position. Say what you did to understand their constraint. Say what you conceded. Say
what you held, and why that thing was non-negotiable. End with the working relationship
afterwards, because interviewers are listening for whether the partnership survived.

*Your material:* story 3 or 4. The evaluation-bar version is in section 4, and it is a
stronger version of this question than a generic scheduling conflict.

*The trap:* telling a story where you were entirely right. Prepare the part where their
objection was reasonable.

### "Tell me about a decision you made with incomplete information."

*Graded on:* whether you knew what you did not know, and whether you bought information
before committing.

*Structure.* State the decision and the deadline that forced it. Name the unknown. Say what
cheap experiment you ran to shrink it, and what you decided to live with. Give the outcome,
including whether the bet was right. Say what you would have measured with another week.

*Your material:* story 2. The oracle test (hand the classifier a ground-truth legend and
measure the ceiling) is exactly "the cheap experiment that shrank the unknown", and the
decision rule is what you committed to before the results existed.

### "Tell me about a project that failed."

*Graded on:* whether you can pick a real failure, and whether you own the part that was
yours.

*Structure.* Say what the goal was and that it was not met. Say what you personally got
wrong, first, before any external factor. Then the external factors, briefly. Then the
detection: how long before you knew, and what that says about the monitoring you designed.
Then the change you made afterwards, in the system or in how you work.

*Your material:* story 1 in its full form (section 5). If the interviewer wants a
project-level failure rather than a regression, **[FILL: a project that did not ship or did
not achieve its goal. If you do not have one, say so precisely: "no project I led was
cancelled; the closest is X, where I got Y wrong and here is what it cost."]**

*The trap:* a failure caused entirely by other people, or a failure with no cost.

### "Tell me about influencing without authority."

*Graded on:* whether other engineers change what they build because of you.

*Structure.* Name the thing you wanted that you could not mandate. Say who had to agree and
what their objection was. Say what you built or measured to make the argument (the artifact
matters more than the persuasion). Say what you gave up to get agreement. Then the adoption
evidence, which is the part candidates forget: who used it, for how long, and whether it
outlived your involvement.

*Your material:* story 4. Reading the OCR column into the classifier creates a dependency on
another team's output, which means agreeing on a contract for that column, and that
conversation is an influence story with a technical artifact behind it. **[FILL: who owned
the OCR column, what they were worried about, and what you agreed to.]**

### "Tell me about mentoring someone."

*Graded on:* whether your effect on other people is specific or vague.

*Structure.* One named person (anonymised). What they could not do before. What you actually
did, in concrete acts: pairing sessions, a design review you made them run, a piece of your
own work you handed over. What changed, measured by what they now do without you. What you
got wrong as a mentor, because mentoring answers are where candidates sound rehearsed.

*Your material:* story 7. **[FILL: this one is entirely yours.]**

### "Tell me about disagreeing with your manager."

*Graded on:* backbone, and then whether you can commit after losing.

*Structure.* The disagreement in one sentence, with their position stated fairly. How you
made the case (data, a prototype, a written doc). The decision. If you lost: what you did
next, and whether the outcome proved you right or wrong. Interviewers value a clean
"I disagreed, I lost, I executed, and the result was that they were right" more than a story
where you won.

*Your material:* story 8. **[FILL.]**

### "Tell me about prioritising under a hard deadline."

*Graded on:* whether you cut scope explicitly or worked longer hours.

*Structure.* The deadline and what made it immovable. The full scope. What you cut, named
specifically, and who you told. The result. What the cut cost later, because every real
descope has a bill that arrives after the launch.

*Your material:* story 5, the 6-week backfill window over 6 PB, finished in 4.2 weeks.
**[FILL: what you descoped to make that window, and who you had to tell.]**

### "Tell me about a production incident."

*Graded on:* detection, communication and prevention, in that order. The fix is the least
interesting part.

*Structure.* What broke and who noticed. How long between the breakage and the detection
(this number is the one they want). What you did in the first hour: mitigation before
diagnosis. The root cause. The prevention, and whether it was a code change or a process
change. What monitoring gap the incident revealed.

*Your material:* story 1. The detection question is the sharp one: **[FILL: how the two
regressions were detected, how long after rollout, and by whom. If a per-class dashboard
caught them, say so; that is a strong answer, because you designed that dashboard.]**

### "Tell me about saying no to a stakeholder."

*Graded on:* whether you can protect a system without becoming an obstacle.

*Structure.* What was asked and why it was reasonable to ask. What it would have cost, in
terms the stakeholder cares about. What you offered instead. Where it ended up. A good "no"
story contains a counter-offer; a bad one contains only a refusal.

*Your material:* story 2 or 3. The EU non-regression bound is the cleanest version: someone
wanted the North American gain sooner, and the bound was non-negotiable because EU revenue
was already booked against it. **[FILL: who pushed, what they wanted, and how it resolved.]**

## 4. The ML-specific questions generic guides miss

These are the ones where staff candidates separate, because the answers require judgment that
only comes from having shipped a model into a system with consequences.

### "Tell me about a model you shipped that caused harm or a regression."

*Graded on:* whether you can describe a failure you caused with mechanism, bound and
mitigation, without either minimising it or performing contrition.

*What a weak answer sounds like:* "the model had some issues with edge cases and we
retrained". No mechanism, no number, no ownership.

*Your material:* story 1, worked in full in section 5. It is unusually strong material
because both regressions have a diagnosis, a bound, a mitigation and a residual, and because
they share one root cause you can state as a general rule.

### "How did you decide a model was ready to ship?"

*Graded on:* whether "ready" is a defined bar or a feeling.

*Structure.* The bar, written down in advance: the primary metric with its floor, the
guardrails that could block regardless of the primary, and who had authority to say no. Then
the evidence you collected against the bar: offline per-slice, seed variance, an online
experiment, a shadow or staged rollout. Then the thing you shipped with, knowingly: every
launch has a known open risk, and naming yours is the staff move.

*Your material:* StreetSmart. Primary was per-class recall against the customer floor,
secondary was the EU non-regression bound within seed variance, tertiary were precision,
latency and cost. **[FILL: the rollout mechanics: staged by region or by traffic fraction,
who reviewed and approved, and what the rollback criterion was.]**

### "Tell me about a time you killed your own project."

*Graded on:* whether you can stop sunk cost, and whether you would tell your management chain
bad news early.

*Structure.* What you believed at the start and why it was reasonable. The evidence that
changed your mind, stated as a specific measurement. When you told people, relative to when
you knew. What you salvaged.

*Your material:* option A and option B in the StreetSmart options table are paths you killed
before investing, which is a weaker version of this question ("killed before starting" reads
as good scoping rather than as intellectual honesty under sunk cost). **[FILL: something you
killed after investing in it. Candidates from either project: an architecture you prototyped
and abandoned, a fusion variant you championed before the ablation, a feature you built that
did not earn its cost. If you have not killed a project after significant investment, say
that plainly and describe the closest case.]**

### "Tell me about a time the metric and the product disagreed."

*Graded on:* whether you notice when the metric has stopped measuring the thing.

*Structure.* The metric, what it was a proxy for, and the moment the two came apart. How you
noticed. What you did: changed the metric, added a guardrail, or shipped against the metric
with eyes open. What the product outcome was.

*Your material:* StreetSmart has two candidates. The aggregate-against-per-class case: an
aggregate accuracy number would have looked healthy while the trucking cluster sat at 62.7%
recall, which is the argument for per-class evaluation. And the precision-against-recall
case: a recall floor drives the contract, while a precision failure is what actually reaches
a driver, so the guardrail and the objective point in different directions.

### "Tell me about a labelling-quality dispute."

*Graded on:* whether you treat labels as data with their own error process.

*Structure.* The symptom (a class that will not improve, a disagreement between two eval
sets, a model that outperforms its labels on inspection). The measurement: inter-annotator
agreement on a sample, an adjudication pass, a re-labelling of the confusion pairs. The
finding: ambiguous guidelines, a taxonomy that does not cut reality at the joints, or genuine
annotator error. The fix, and its cost in time.

*Your material:* the four failure clusters came out of an error-pattern audit, and
text-discriminated classes are exactly where label guidelines get ambiguous (is a sign with
an unreadable legend the parent class or the specific class?). **[FILL: did you hit a
labelling dispute on the MUTCD taxonomy, and how was it resolved? If the guideline question
above came up, that is the story.]**

### "Tell me about setting an evaluation bar with a partner team who wanted a different one."

*Graded on:* whether you can negotiate a measurement, which is the most consequential
negotiation an ML engineer has.

*Structure.* What each side wanted and why each was rational. The failure mode each bar would
have allowed. What you agreed: usually a primary metric from one side and a guardrail from
the other. How the disagreement showed up later, and whether the agreed bar caught it.

*Your material:* the HD-map partners' contracted per-class recall floor against an internal
preference for aggregate accuracy, and the audit requirement for per-token attention maps,
which made interpretability a shipping criterion alongside accuracy. **[FILL: who wanted what,
and what the negotiation actually was.]**

## 5. One fully worked answer

The question: **"Tell me about a time your model caused a regression in production."**

Lead with this story. Interviewers are used to candidates protecting their failures, and a
candidate who opens the failure drawer themselves, with numbers, changes the tone of the
round. Two to three minutes, then stop.

!!! example "The answer, as you would say it"
    "The clearest one is from StreetSmart. I led a change that added a text modality to our
    sign classifier: we fused the OCR transcription that was already being computed on every
    crop into the classifier through a cross-attention head. It worked, and the North
    American regulatory clusters gained between 17 and 30 points of recall. It also caused
    two regressions, and both were mine.

    The first was on pictogram-strong classes. On something like a generic stop sign, where
    the pixels are already sufficient, OCR sometimes returns garbage: ST0P with a zero, 5T0P
    with a five, at confidence below 0.3. The fused model lost about 1.1 points of recall on
    those classes, because the cross-attention head attended to the text token anyway. When I
    looked at why, the cause was a design choice I had made: I passed the OCR string to the
    text encoder and did not pass the OCR confidence. The model had no way to learn when to
    distrust the text, so it learned the average reliability of the training distribution and
    applied it everywhere.

    The fix had two parts. We passed per-token OCR confidence into the text encoder as an
    input, so the head could learn a reliability gate, and we used the gated-fusion variant
    from our ablation as a re-ranker when confidence was low, which gives a fallback that
    behaves like the old vision-only model. The residual regression was 0.2 points, inside
    seed variance.

    The second regression was subtler and I like it less. Speed-limit precision dropped about
    0.6 points on captures with low light or motion blur, where the recogniser confuses 25
    with 55 or 35 with 85. The text branch reported high confidence on the wrong digit, so
    the confidence input did not save us. We mitigated by calibrating the text branch at a
    higher temperature than the vision branch at inference, which is a blunt instrument. The
    real fix, which I did not get to, is a digit-specific confidence head trained on our
    capture distribution.

    What I took from it: when you fuse a learned upstream signal, its confidence is part of
    the signal. I had treated OCR as reliable because it was accurate on average, and average
    accuracy is the wrong summary when the failures are systematic and correlated with the
    cases you care about. I now ask, for any upstream model feeding mine, what its error
    process looks like on my hard slice, and whether my model can see when to distrust it."

**Why this answer works, move by move.**

It opens with the win in one sentence and then hands over the failure voluntarily. The
interviewer does not have to dig, which means the rest of the round is spent on your
reasoning instead of on their excavation.

It attributes the cause to a decision you made, by name: the confidence value existed on the
row and you left it out. Owning a *decision* is stronger than owning an *outcome*, because
decisions are what you will make again at their company.

It bounds the damage with numbers on both sides: 1.1 points lost on one cluster against 17 to
30 points gained on five, and a residual of 0.2 inside seed variance. Bounding is what
separates ownership from self-flagellation.

It includes a second regression that is *still not fully fixed*, and says so. Candidates
almost never do this, and it is the most credible moment in the answer.

It ends on a rule stated generally enough to apply to a system the interviewer owns, which is
what makes the story evidence about your future behaviour instead of a report about your past.

**The follow-ups, and how to take them.**

*"How did you find out?"* Answer with the detection path and the time. **[FILL: per-class
dashboard, a slice alert, a partner report, or a manual review, and how long after rollout.]**
If the detection was slow, say so and say what you changed; a slow detection owned is better
than a fast detection claimed.

*"Why did you not catch it before launch?"* The offline evaluation set under-represented the
noisy-OCR condition. The general answer: an evaluation set sampled from the average
distribution cannot measure a failure that is conditional on an upstream model being wrong,
so the eval needs a slice defined by upstream confidence. That is a specific, checkable
lesson.

*"What did it cost?"* Be precise about the blast radius: which classes, for how long, and
what the downstream effect was. **[FILL: whether the regression reached partners or was
caught in a staged rollout.]**

*"Would you have shipped it again?"* "Yes, with the confidence input in version one. The
fusion itself was correct; the reliability modelling was late. If the trade had been a 1.1
point loss on a contracted class against a gain on a non-contracted one, the answer would be
no, and that is the calculation I would run first now."

## 6. Amazon-style leadership-principle framing

Amazon's loop is the most explicit public example of behavioral grading: four to six
interviews, each interviewer assigned specific Leadership Principles, plus a Bar Raiser from
outside the hiring team, with STAR as the recommended answer structure
([Amazon: the interview loop](https://amazon.jobs/content/en/how-we-hire/interview-loop);
[Leadership Principles](https://amazon.jobs/content/en/our-workplace/leadership-principles);
[About Amazon: the interview process](https://www.aboutamazon.com/news/workplace/amazon-interview-guide)).
Other companies grade similar dimensions with less ceremony, so preparing this way is not
wasted effort elsewhere.

Four things about the format that change how you prepare.

**Each interviewer is assigned principles, so stories repeat across the day.** Prepare your
eight stories with two angles each, and vary which part you emphasise. Repeating the same
story to two interviewers is fine; repeating it identically is not, because they compare
notes.

**Dive Deep means numbers.** A principle named Dive Deep produces questions like "what was
the P95, and how was it measured?" mid-story. Your StreetSmart numbers are the asset here:
know the denominators.

**Ownership means the boundary of what was yours.** Expect "what specifically did you do?"
two or three times in one answer. Answer it each time without irritation and without
inflating.

**Have Backbone; Disagree and Commit is one principle with two halves.** A story where you
disagreed and won covers only the first half. The strongest version has you disagreeing,
losing, committing fully and reporting the outcome honestly.

A mapping to start from. Fill the empty cells before an Amazon loop.

| Principle | Your story | The angle |
|---|---|---|
| Customer Obsession | the contracted recall floor and the EU non-regression bound | the metric came from what the customer was paying for |
| Ownership | the OCR-reliability regressions | you caused it, you found it, you bounded it |
| Invent and Simplify | reusing the OCR column already on the Spanner row | the simplification is the invention |
| Are Right, A Lot | the decision rule, and the oracle test that justified it | you bet before the results and said what would change your mind |
| Learn and Be Curious | the error-pattern audit that produced the four clusters | you went and looked instead of retraining |
| Hire and Develop the Best | **[FILL: story 7]** | |
| Insist on the Highest Standards | per-class per-region evaluation over aggregate accuracy | you made the bar harder when it would have been easier not to |
| Think Big | the fallback path for the tail, and transferring the design to new countries | |
| Bias for Action | shipping with frozen backbones to hold the 6-week window | you took the constrained path to get a result this cycle |
| Frugality | marginal compute near zero, 1.8 ms P95, no new dependency | |
| Earn Trust | leading with the regressions | |
| Dive Deep | the ablation and the per-cluster table | |
| Have Backbone; Disagree and Commit | **[FILL: story 8]** | |
| Deliver Results | 4.2 weeks over 6 PB, inside the window | |

## 7. Delivery mechanics

**Length.** Two to three minutes. Interviewers have three or four questions to get through
and a rubric to fill, and an answer past four minutes starts costing you the questions you
were going to be strong on.

**"I" and "we".** Use "we" for the team's work and "I" for your decisions, and switch
deliberately: "we ran the ablation; I designed it to control for parameter count." A staff
candidate who says "we" throughout gets levelled down, and a candidate who says "I" about
everything gets flagged as someone who does not credit their team. The mix is the signal.

**When you do not have the story.** Say so quickly, then offer the nearest real thing. "I
have not shut down a project after a large investment. The closest is where I killed my own
preferred fusion variant after the ablation contradicted me; here is what that cost." An
honest near-miss beats a stretched story, because a stretched story collapses at the second
follow-up.

**Confidential numbers.** Decide in advance which figures you will give. "I can give you the
shape and the ratio, not the absolute" is a normal and respected answer, and giving ratios
("about a quarter of our volume") keeps the story concrete.

**Notes.** Bring one page with your eight story titles and their numbers. Glancing at a list
of titles is fine and reads as prepared. Reading an answer does not.

**The question at the end.** Ask something that only this interviewer can answer: what the
team's evaluation practice looks like, how a model gets to production, what the last
regression was and how it was found. Interviewers remember candidates who ask about failure
modes, because that is what the job is.

## Exercises

**★ 1. Inventory audit.** Fill the eight-story table in section 2. Mark each story with
whether you have a number, a rejected alternative, and a cost. Any story missing two of the
three is not ready.

??? success "Solution"
    The usual pattern: project stories have numbers and alternatives but no stated cost, and
    people stories have costs but no numbers. For a people story, the number can be the
    duration or the scope ("three months of weekly pairing", "she now runs the review I used
    to run"), which is enough to make the claim checkable.

**★★ 2. The "what did you personally do" drill.** Have someone interrupt each of your answers
twice with "what specifically was your part in that?" Answer without repeating yourself and
without inflating.

??? success "Solution"
    The technique that works: pre-split each story into three ownership layers, what you
    decided, what you built, what you reviewed or influenced. Each interruption gets the next
    layer. When you run out of layers, say so: "past that, the implementation was X's, and my
    part was the interface and the eval it had to pass."

**★★★ 3. The regression story, timed and recorded.** Deliver the section 5 answer from
memory in under three minutes, then answer the four follow-ups cold. Score yourself on
whether you named the decision, gave both numbers, admitted the unfixed part, and ended on a
general rule.

??? success "Solution"
    The part that usually goes missing under time pressure is the bound: candidates give the
    loss without the gain it sat against, which makes the story sound worse than it was, or
    the gain without the loss, which makes it sound rehearsed. Say both, in one sentence,
    early: "1.1 points on one cluster, against 17 to 30 across five."

## References

* Amazon Jobs: [Leadership Principles](https://amazon.jobs/content/en/our-workplace/leadership-principles),
  [The interview loop](https://amazon.jobs/content/en/how-we-hire/interview-loop).
* About Amazon: [Your complete guide to the Amazon interview process](https://www.aboutamazon.com/news/workplace/amazon-interview-guide).
* Google Careers: [How we hire](https://www.google.com/about/careers/applications/how-we-hire).
* This book: [what staff-level signal looks like](../preface/interview-signal.md) for the
  round-by-round rubric, [the CARL framework](01-carl-framework.md) for the presented format,
  and [the StreetSmart deep dive](02-streetsmart-deep-dive.md) for the source material behind
  most of the stories above.
