# Part XX: your projects

> **Why this matters at staff level.** At staff level the project deep dive is often the
> highest-weighted round in the loop, because it is the only one where the interviewer
> grades work you actually shipped rather than work you improvised at a whiteboard. It is
> also the round candidates under-prepare, on the theory that they already know their own
> project. You know the project. You have almost certainly never argued it against a
> hostile clock to someone who was not there.

Every other part of this book teaches material that is true for everyone. This part is
about your material: the StreetSmart sign detection and classification work at Google
Maps (Q4 2024 to Q2 2025) and the radar UAV edge ML work at Epirus. The goal is to turn
both into answers you can deliver cold, at three lengths, with the numbers attached.

## The two formats

The same project gets asked about in two very different rounds, and they reward opposite
instincts.

**Format 1: the presented deep dive.** A 45 or 60 minute slot, usually with slides, often
with two or three people in the room, sometimes recorded for the debrief. You drive. You
are expected to arrive with structure, to hold the room, and to survive interruption.
Google, Meta, Apple and most autonomy companies run some version of this for staff and
above; frontier labs run it as a "walk us through something you built" session with a
whiteboard instead of slides. The deck for StreetSmart in
[chapter 02](02-streetsmart-deep-dive.md) is built for this.

**Format 2: the conversational project question.** Five to twelve minutes inside a
behavioral or a hiring-manager round, prompted with "tell me about a project you are
proud of" or "walk me through something technically hard you did recently". No slides, no
diagram, and the interviewer will cut you off at the first sign of a timeline narration.
You need a 90-second version and a 5-minute version that both stand alone, and you need
to choose which project answers the question that was actually asked.

| | Presented deep dive | Conversational project question |
|---|---|---|
| Length | 45 or 60 min, you own 30 to 40 of them | 5 to 12 min inside another round |
| Artifact | 10 to 14 slides, two diagrams | your voice, maybe a whiteboard box diagram |
| Graded on | technical depth, decision quality, evaluation design, how you handle challenge | scope of ownership, judgment, honesty, whether you can compress |
| Fails when | you narrate a timeline, you never state the alternative you rejected | you go 9 minutes without saying what you decided |
| Opens with | a 90-second TL;DR, then context | one sentence of outcome, then "want the technical path or the people path?" |
| Ends with | learnings and what you would do next | the number, and one thing you got wrong |

In the presented format, the interviewer is grading whether they would trust you to lead a
project of that size. Specifically: can you state a constraint precisely, did you consider
the option they would have considered, do you know why your numbers are what they are, and
what happens to your composure when someone doubts the result. In the conversational
format they are grading scope and truthfulness: how much of this was yours, and does the
story survive two follow-up questions.

## Which of your projects answers which question

Pick by the competency being probed, not by which project you like more.

| The question, as they ask it | Best source project | What you lead with |
|---|---|---|
| "Tell me about multimodal fusion you have done." | StreetSmart | frozen ViT plus frozen BERT, cross-attention head, the ablation table |
| "Have you dealt with distribution shift?" | StreetSmart | Vienna Convention pictograms vs MUTCD text-first signs, per-cluster recall collapse |
| "How do you design an evaluation?" | StreetSmart | per-class per-region slices, the EU non-regression bound, read by row |
| "When would you freeze a backbone?" | StreetSmart | zero EU regression by construction, head-only backfill over 6 PB |
| "Design under a cost constraint." | StreetSmart | 1.4B crops/day, P95 280 ms, marginal compute near zero by reusing OCR |
| "Tell me about an auditable or regulated system." | StreetSmart | per-token attention maps for partner audit, why option A was not auditable |
| "Tell me about a regression you caused." | StreetSmart | the two OCR-reliability regressions, diagnosed and bounded |
| "Tell me about quantization you have shipped." | Epirus | INT8 PTQ and QAT on a radar detector, what you checked after quantizing |
| "Have you deployed on embedded hardware?" | Epirus | the latency and power budget, depthwise-separable backbone |
| "Tell me about a hardware or physics constraint." | Epirus | radar resolution limits, what the sensor gives you that a camera cannot |
| "Tell me about a time you had incomplete information." | either | pick the one with the sharper decision, then say what you would have measured |
| "What would you do with another quarter?" | StreetSmart | the digit-specific OCR confidence head, the OOD fallback for the tail |

Two projects covering twelve prompts is enough. Interviewers get suspicious when a
candidate has a fresh project for every question; they get reassured when two projects
are known to the bone.

## What this part contains

* [The CARL framework](01-carl-framework.md). Context, Actions, Results, Learnings as a
  deep-dive structure: why it fits an ML project better than STAR, the minute-by-minute
  time budget for both slot lengths, slide mechanics, how to draw a diagram incrementally,
  how to take an interruption without losing your thread, and a rubric for scoring your
  own rehearsal.
* [The StreetSmart deep dive](02-streetsmart-deep-dive.md). The worked example, with the
  TL;DR written out as a script, every section of the deck expanded into what you say and
  why, and twenty-two probes an interviewer will push on with model answers.
* [The edge perception deep dive](03-edge-perception-deep-dive.md). A scaffold for the
  Epirus project, with the quantization and efficient-convolution material you must be
  ready to defend and clearly marked blanks where only you have the facts.
* [Behavioral and leadership](04-behavioral-leadership.md). The question bank, the
  ML-specific behavioral questions that generic guides skip, one fully worked answer from
  the StreetSmart regressions, and Amazon-style leadership-principle framing.

## Preparation checklist

Work top to bottom. Nothing below the line about rehearsal counts until the things above
it are done.

- [ ] Write the 90-second TL;DR verbatim and time it. Under 100 seconds or cut it.
- [ ] Reconstruct every number from the source of truth (dashboards, design docs, launch
      reviews) and write each one on the slide where you will say it. Numbers you cannot
      source do not go in the deck.
- [ ] For each number, prepare the question "how did you measure that?" Sample size, slice
      definition, seed variance, date range.
- [ ] Write the options table: every path you evaluated, and the single sentence that
      disqualified each rejected one.
- [ ] Write the decision rule that ordered those options before accuracy entered the
      argument. This is the sentence the interviewer will remember.
- [ ] Draw both diagrams from memory on paper. If you cannot, you do not know them well
      enough to draw them live.
- [ ] List the three things you got wrong, with the diagnosis and the bound on the damage.
- [ ] Write the counterfactual for the headline result: what would have happened with no
      project, and how you know.
- [ ] Prepare the credit map: for each major decision, who proposed it, who built it, what
      you personally owned. Vague credit is the fastest way to lose a staff hire.
- [ ] Anonymise anything confidential. Decide in advance which numbers you will not give
      and what you will say instead ("I can give you the shape of it, not the absolute").
- [ ] Rehearse aloud, standing, with a timer.
- [ ] Rehearse being interrupted: have someone stop you at minute 6 with "why not just
      fine-tune?" and continue without restarting.
- [ ] Rehearse the 5-minute conversational version and the 90-second version separately.
      They are different performances, not truncations.

## A two-week rehearsal schedule

| Day | Work | Output |
|---|---|---|
| 14, 13 | Reconstruct the facts. Pull the numbers, re-read the design doc and launch review. | One page of verified facts per project. |
| 12 | Write the options table and the decision rule. | The Actions skeleton. |
| 11 | Build the deck to 12 slides. One claim per slide, headline as a sentence. | Draft deck. |
| 10 | Draw both diagrams from memory, then rebuild them as the incremental build order you will narrate. | Diagram build order, on paper. |
| 9 | Write the 90-second TL;DR and the 5-minute version. Time both. | Two scripts. |
| 8 | Full run-through alone, timed, out loud, no audience. | Where you ran long. |
| 7 | Cut 20% of the deck. Whatever survived rehearsal without earning its minute goes. | 10 to 12 slides. |
| 6 | Answer the 22 probes in [chapter 02](02-streetsmart-deep-dive.md) out loud, cold, one pass, no notes. | List of the ones you fumbled. |
| 5 | Repair the fumbles: go find the real answer, or decide what you will say when you do not have it. | Repaired answers. |
| 4 | Mock with an engineer who knows ML but not your domain. Ask them to interrupt three times. | Their questions, written down. |
| 3 | Mock with someone senior who will push on credit and on the weakest number. | The two weakest points in the story. |
| 2 | Epirus scaffold: fill the blanks in [chapter 03](03-edge-perception-deep-dive.md), rehearse the 10-minute version. | Second project ready. |
| 1 | Behavioral stories from [chapter 04](04-behavioral-leadership.md), five of them, out loud once each. Then stop. | Done. Sleep. |

The day before an interview, rehearsal has diminishing returns and rising anxiety. One
pass, then close the laptop.

## How this part connects to the rest of the book

The deep dive is the round where the rest of the book becomes evidence. When you explain
why cross-attention beat gated fusion, you are teaching
[attention mathematics](../part05-sequence-transformers/03-attention-mathematics.md) with
your own ablation attached. When you explain the EU non-regression bound, you are doing
[evaluation design](../part13-retrieval-eval-reliability/02-evaluation.md) with a customer
contract behind it. The generic version of each argument lives in those chapters; this
part is about the version with your name on it.

Read [what staff-level signal looks like](../preface/interview-signal.md) first if you have
not. It defines the senior-to-staff gradient that every rubric in this part scores against.
