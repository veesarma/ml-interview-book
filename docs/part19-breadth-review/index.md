# Part XIX: Breadth Review

> **Why this matters at staff level.** In a 45-minute ML depth round you will be asked
> twelve questions, and you have about three minutes for each. Parts I to XVIII teach
> you the material properly; this part is the recall layer on top of them, written as
> the questions are actually asked and answered at the length they are actually
> answered. An interviewer is sampling your breadth to find a seam worth pushing on.
> Every answer here is built so that the first sentence is a complete answer and the
> rest earns it.

## What this part is

Every question in Part XIX has two layers.

**The answer** is the core technically correct response: the definition, the mechanism,
the derivation if there is one. It is what a strong L5 candidate says, and it is graded
as correct.

**The staff move** is what an L6 candidate adds without being asked: the trade-off
named as a trade-off, the failure mode, the business contingency, or the thing the
interviewer was hoping you would surface. It is usually one or two sentences, and it is
where the level distinction actually gets made.

The two layers are separated on purpose. When you drill these, check both. Most
candidates who fail a depth round fail it while answering every question correctly.

## How this differs from the deep chapters

| | Parts I to XVIII | Part XIX |
|---|---|---|
| Unit | A topic | A question |
| Length | 2,500 to 6,000 words plus code | 150 to 500 words |
| Goal | You can teach it | You can answer it out loud in three minutes |
| Contains | Derivations, implementations, systems analysis, case studies | The claim, the mechanism, the trade-off, the failure mode |
| When to read | Months out, once, carefully | Weekly from month two, daily in the last week |

Every question here carries a link to the chapter that teaches it properly. A question
you cannot answer is a pointer into the book, which is the only use of a
self-test that matters.

## The meta-skill being tested

Interviewers at this level assume you know what cross-entropy is. They are checking
three other things.

**Do you reason from first principles or from memorised facts?** The follow-up exists to
find this out. When they ask "why $1/\sqrt{d_k}$?", the candidate who memorised says
"to stop the logits getting large" a second time, louder. The candidate who understands
says "the dot product of two $d_k$-dimensional unit-variance vectors has variance
$d_k$, so the logits have standard deviation $\sqrt{d_k}$, softmax saturates and its
gradient goes to zero, so we divide to hold the logit variance at 1."

**Do you think in trade-offs and then commit?** Almost every good answer at this level
begins "it depends", and almost every bad answer ends there. Make the dependency
explicit, then state a decision under a stated business context: "given a 30 ms
on-device budget I would take the int8 quantised CNN and accept the 1 point of mAP,
and here is the slice I would watch to catch the regression."

**Do you know where the bodies are buried?** Failure modes, sharp edges, the distance
between the paper and the deployment. Surfacing one unprompted is the highest-signal
move available to you, because reading a paper does not teach you that training and
serving compute features differently at 3am.

## The reusable answer skeleton

When an open-ended question lands, say the headline, then walk this out loud. The
ordering signals seniority by itself.

1. **Define the term precisely.** Ten seconds. It also buys you thinking time and
   catches the case where you and the interviewer mean different things.
2. **Give the standard answer.** The mechanism, with the equation if there is one.
3. **Name the trade-off.** Which axis are you spending: quality, latency, memory,
   compute, robustness, maintainability?
4. **Say where it breaks.** The failure mode you have personally seen or can describe
   concretely.
5. **Commit, given a business context.** "For a fraud queue with 2,000 analyst-hours a
   week, I would optimise precision at that queue depth and hold recall at the current
   level."

Steps 3 to 5 are the part most candidates skip. They are the part being graded.

## The chapters, and what they summarise

| Chapter | Questions about | Summarises |
|---|---|---|
| [1. Basics, losses, models](01-basics-losses-models.md) | Bias and variance, non-IID data, overfitting diagnosis, cross-entropy vs MSE, L1 and L2 as priors, ridge vs lasso, hinge vs log loss, softmax vs OvR vs OvO, linear models and MLPs, feature scaling, CART, bagging, boosting | [Part I](../part01-math/index.md), [Part II](../part02-classical/index.md), [Part III](../part03-neural-nets/index.md) |
| [2. Evaluation and debugging](02-evaluation-debugging.md) | Splits and CV, optimistic CV, AUC against the business KPI, calibration, label noise, missing data, imbalance, leakage, and the five debugging questions | [Part XIII](../part13-retrieval-eval-reliability/index.md), [Part XVII](../part17-ml-system-design/index.md) |
| [3. Deep learning and Transformers](03-deep-learning-transformers.md) | Non-linearity, init, backprop, optimizers, schedules, dropout, the three normalizations, then attention, $1/\sqrt{d_k}$, heads, masking, positions, SwiGLU, pre-norm, encoder vs decoder, tokenization, scaling | [Part III](../part03-neural-nets/index.md), [Part V](../part05-sequence-transformers/index.md), [Part VI](../part06-llm-training/index.md) |
| [4. LLMs and multimodal](04-llms-multimodal.md) | MHA to MQA to GQA to MLA, the standard LLM recipe, RLHF and DPO, LoRA, RAG, MoE, reasoning models, CNNs to ViT, detection, segmentation and SAM, CLIP and VLMs, self-supervision in vision | [Part VI](../part06-llm-training/index.md), [Part VII](../part07-post-training/index.md), [Part VIII](../part08-multimodal/index.md), [Part IV](../part04-vision/index.md) |
| [5. Serving, data engines, real-world](05-serving-data-engines.md) | Latency, throughput, caching, large-scale training, auto-labeling, active-learning selection, the long tail, sensor fusion | [Part XIV](../part14-systems/index.md), [Part X](../part10-self-supervised/index.md), [Part XI](../part11-perception-autonomy/index.md) |
| [6. Rapid-fire derivations](06-rapid-fire.md) | The thirty things you should be able to produce on a whiteboard without hesitation, plus delivery | All of the above |

## How to drill these as flashcards

The questions are in collapsible blocks so the page is a self-test by default. A
drill pass looks like this.

* **Read the question heading only.** Do not open the block.
* **Answer out loud, standing up, with a timer.** Three minutes. Out loud matters,
  because the failure mode in interviews is knowing the answer and assembling it
  badly. Silent reading hides that completely.
* **Open the block. Grade yourself on two axes.** Did you get the answer? Did you get
  the staff move? Score 0, 1 or 2.
* **Anything below 2 goes in the review pile.** Anything scoring 0 goes to the linked
  deep chapter tonight, not to a re-read of this page.
* **Re-drill the pile at one day, three days, and one week.** Spacing is what moves
  these into recall you can reach under pressure.

A pass over one chapter takes 40 to 60 minutes cold and 15 minutes warm. Do not read
the answers straight through, because reading breeds the feeling of knowing without
the ability to produce.

## One week out

Seven days, about 90 minutes a day, assuming you have already worked through the deep
chapters at least once. The order front-loads the two chapters interviewers weight
most heavily and leaves the last two days for delivery instead of content.

| Day | Morning (45 min) | Evening (45 min) |
|---|---|---|
| **7** | [Chapter 2](02-evaluation-debugging.md), the five debugging questions, out loud, timed | Re-derive the AUC and calibration answers at a whiteboard; open the figure and read your own operating point off it |
| **6** | [Chapter 3](03-deep-learning-transformers.md) §attention through §positional encodings | Write scaled dot-product attention with a causal mask from memory, then check against [attention mathematics](../part05-sequence-transformers/03-attention-mathematics.md) |
| **5** | [Chapter 4](04-llms-multimodal.md) LLM half (MQA to MLA, RLHF and DPO, LoRA, RAG, MoE) | [Chapter 4](04-llms-multimodal.md) vision half (ViT, detection, SAM, CLIP, VLMs) |
| **4** | [Chapter 1](01-basics-losses-models.md) in full | [Chapter 5](05-serving-data-engines.md) in full |
| **3** | [Chapter 6](06-rapid-fire.md), every derivation on paper, no notes | Re-drill everything you scored 0 or 1 on so far |
| **2** | Mock: 12 random questions drawn across all six chapters, 3 min each, recorded | Listen back to the recording. You are grading headline-first delivery, not content |
| **1** | The review pile, once | Stop. Re-read the [TL;DR cards](../preface/how-to-use.md) of the two chapters closest to the role and sleep |

On day 2, the recording is the exercise. Almost everyone discovers they bury the
headline, hedge past the point of usefulness, or answer a question the interviewer did
not ask. Those are all cheap to fix in a week and expensive to discover in the room.

## A note on what has been corrected

This part was built from a working review document, and where that document carried a
claim that is dated, folklore, or imprecise, the text says so in a note rather than
repeating it. The cases are flagged inline: the tie term in the AUC identity, the
Gini-versus-entropy folklore, inverted dropout, what pre-norm actually buys you with
respect to warmup, the token-to-parameter ratio after Chinchilla, and the status of
LoRA's zero-initialised $B$.
