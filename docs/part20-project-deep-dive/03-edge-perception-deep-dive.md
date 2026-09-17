# Worked deep dive: radar UAV perception at the edge

> **Why this matters at staff level.** Your second project is what makes the first one
> credible. A candidate with one deep project looks like someone who got lucky with an
> assignment; a candidate with a second project in a different regime (embedded hardware,
> a power budget, a non-optical sensor) looks like an engineer whose judgment transfers.
> This chapter is a scaffold you complete, because the facts belong to you.
> Everything technical here is material you will be asked to defend once you claim INT8
> quantization and depthwise-separable backbones on your résumé.

## TL;DR: the interview card

* What the project gives you in an interview: quantization you actually shipped (PTQ and
  QAT), an embedded latency and power budget, and a sensor that is not a camera.
* PTQ costs an afternoon and no labels; QAT costs a training run and buys back most of the
  remaining accuracy. The published gap for INT8 on convolutional networks is about 2% of
  float for per-channel PTQ, narrowing to about 1% with QAT
  ([Krishnamoorthi 2018](https://arxiv.org/abs/1806.08342)).
* INT8 error has two sources that trade against each other: clipping (range too narrow) and
  rounding (range too wide). Everything in calibration is choosing between them.
* Per-channel weight quantization exists because depthwise convolutions have wildly
  different per-channel ranges, and per-tensor scaling destroys the small ones. Depthwise
  networks are the canonical PTQ failure case.
* Depthwise-separable convolution cuts multiply-accumulates by $\tfrac{1}{C_{out}} +
  \tfrac{1}{K^2}$, about 8 to 9 times for $K=3$, and delivers less than that in wall-clock
  because the depthwise stage is memory-bound.
* Radar gives you range and radial velocity directly, at night, in dust and fog. It gives
  you almost no angular detail, so classification leans on Doppler signatures
  instead of shape.
* Range resolution is $\Delta R = c / (2B)$: the bandwidth sets it, not the model.
* Your delivery target: a 10-minute version with one constraint, one decision, one number
  and one failure. It does not need to be a 40-minute deck.

## How to use this chapter

You have a project you did not document for an interview, so the facts are thin in memory
and thick in artifacts you may no longer have. Three moves fix that.

First, fill the CARL skeleton in section 1, out loud, with whatever you have. Where you
cannot recall a number, write the shape instead ("the board had single-digit watts, not
tens") and mark it as a shape when you say it.

Second, read sections 2 and 3 and rehearse the technical material until you could teach it
without the project attached. An interviewer who hears "we used INT8 PTQ then QAT" will
follow with "where does the error come from?", and that question is answerable from theory
alone. Strength on the theory buys you credibility on the parts of the project you
half-remember.

Third, use the probe list in section 4 to find what you cannot answer, and decide in advance
which of those you will answer with "I do not remember the number, here is the shape and
here is how we measured it".

Every **[FILL: ...]** below is a blank only you can close.

## 1. The CARL skeleton, with prompts

The structure is the one from [the CARL chapter](01-carl-framework.md), compressed to the
10 to 15 minutes this project will actually get.

### Context

* **The system.** **[FILL: what the system did end to end. Radar sensor, signal processing
  front end, then what? Did your model consume raw ADC data, a range-Doppler map, a
  range-azimuth map, or detections from a classical CFAR stage?]**
* **The mission.** **[FILL: what decision the output drove. Detection and classification of
  small UAVs at what range? Cue for another sensor or an effector? Who or what consumed the
  output, and what did a false positive cost them?]**
* **The platform.** **[FILL: the compute. Which SoC or accelerator, how much memory, what
  power envelope, and whether it was shared with the signal-processing pipeline.]**
* **The budget.** **[FILL: the latency and frame-rate target and where it came from. A
  radar frame rate, a track-update requirement, an operator-latency requirement?]**
* **The baseline.** **[FILL: what existed before. A classical CFAR plus tracker? A float32
  CNN that did not fit? Nothing?]**
* **Your role.** **[FILL: what you owned, what you built, who else was on it.]**

The Context facts that do the most work in an interview are the power envelope and the
latency budget, because they are what make the problem different from a server-side problem.
Lead with them.

### Actions

* **The framing.** **[FILL: what you identified as the binding constraint. Was it latency,
  memory footprint, power, or accuracy at a required range?]**
* **The architecture decision.** Depthwise-separable convolutions in place of standard
  convolutions. **[FILL: what you replaced, how deep the network was, what input resolution,
  and what the parameter and MAC counts were before and after.]** The derivation you owe the
  interviewer is in section 3.2.
* **The quantization decision.** INT8, PTQ first and then QAT. **[FILL: why you went to QAT.
  How much accuracy did PTQ lose, and on what metric? That number is the justification for
  the extra training run, and it is the single most valuable number to recover.]**
* **The rejected alternatives.** **[FILL: at least two. Candidates: a smaller float model, a
  pruned model, 16-bit instead of 8-bit, running the classical detector alone, moving
  compute off-platform.]** Say the disqualifying clause for each.
* **The calibration and training details.** **[FILL: calibration set size and how you chose
  it; range estimator (min-max, percentile, MSE); per-channel or per-tensor; what stayed in
  higher precision.]**

### Results

* **The headline.** **[FILL: latency before and after, in milliseconds on the target board,
  measured how (mean, P95, with or without pre and post-processing).]**
* **The accuracy cost.** **[FILL: the detection metric before and after quantization, per
  range band or per target class if you have it.]**
* **The power number.** **[FILL: watts or joules per frame, if you measured it. If you did
  not, say so; guessing a power figure in front of an embedded engineer ends the round.]**
* **The counterfactual.** **[FILL: what would have happened without the work. Did the float
  model miss the frame budget? By how much?]**

### Learnings

* **[FILL: a transferable rule, stated so it applies to a system the interviewer owns.]**
  Candidates that are almost always true for this kind of work and are worth checking
  against your memory: quantization error concentrates in specific layers instead of
  spreading evenly, so per-layer diagnostics beat end-to-end accuracy for debugging;
  a FLOP reduction is a claim about arithmetic and not about wall-clock, so measure on the
  device; and post-quantization score distributions shift, so every threshold downstream of
  the model has to be re-tuned, never inherited.
* **What you would do differently.** **[FILL: one technical, one process.]**

## 2. Turning a thin memory into a strong ten minutes

Here is the raw material most people have after two years: "I built a small CNN for radar
UAV detection, used depthwise-separable convolutions, quantized it to INT8 with PTQ and then
QAT, and got it inside the latency and power budget on the embedded board."

That is one sentence with no constraint, no alternative, no number and no failure. Four
reconstruction moves turn it into ten minutes, and each move is a question you can answer
from memory of the *situation* even when you have lost the numbers.

**Move 1: recover the constraint by recovering the argument.** What were people worried
about in review? Whoever was nervous was nervous about the binding constraint. If the debate
was about whether the model would fit in memory, the constraint was footprint. If it was
about whether detection range would drop, the constraint was accuracy at range.
**[FILL: which argument did you have?]**

**Move 2: recover the baseline from the decision to act.** Nobody quantizes a model that is
already fast enough. The float model must have missed something. **[FILL: what did it miss,
and by roughly how much? A factor of two? An order of magnitude?]** A ratio you are confident
in beats an absolute number you are not.

**Move 3: recover the alternative from what you rejected in the moment.** You considered
something else first. Most people in this position considered a smaller float network, or
16-bit, before committing to INT8. **[FILL: what did you try first, and what made you stop?]**

**Move 4: recover the failure from what surprised you.** Something did not behave. For
quantized detectors the usual suspects are a single layer that dominates the error, a
confidence-threshold shift that silently changed the operating point, or an accuracy drop
concentrated in one target class or range band. **[FILL: what surprised you?]** If nothing
surprised you, then either the work was routine, in which case say so and pick a different
story, or you have forgotten the surprise, in which case dig.

Assembled, the ten-minute answer has this shape, and you should rehearse it as a script.

!!! example "The shape of the ten-minute version"
    "At Epirus I worked on radar-based UAV perception running on an embedded platform. The
    constraint that drove everything was **[FILL: the constraint]**: we had **[FILL: the
    budget]**, and a straightforward float convolutional detector was **[FILL: the factor]**
    over it.

    I had two levers, architecture and numerics, and I used both. On architecture I replaced
    standard convolutions with depthwise-separable blocks, which for 3 by 3 kernels cuts
    multiply-accumulates by roughly an order of magnitude, though the wall-clock gain is
    smaller because the depthwise stage is memory-bound rather than compute-bound. On
    numerics I went to INT8. I started with post-training quantization because it costs an
    afternoon and no labels, measured **[FILL: the PTQ accuracy loss]**, and that number is
    what justified moving to quantization-aware training, which recovered **[FILL: how
    much]**.

    The detail I would flag is that quantizing a detector is not the same as quantizing a
    classifier. The classification head degrades gracefully; the regression head and the
    confidence distribution do not. After quantization our score distribution shifted, so
    the operating threshold we had tuned in float was no longer the same operating point,
    and **[FILL: what that did to your false-alarm rate]**. Since then I treat every
    threshold downstream of a quantized model as something that has to be re-tuned rather
    than inherited.

    Where it landed: **[FILL: the latency and accuracy numbers]**. And the thing I would do
    differently: **[FILL: one thing]**."

Ten minutes of that, delivered with two numbers you are sure of and the rest offered as
shapes, outperforms twenty minutes of confident detail that falls apart under one follow-up.

## 3. The technical content you must be ready to defend

### 3.1 Quantization

**The mapping.** Affine (asymmetric) quantization of a real value $x$ to $b$ bits uses a
scale $s$ and a zero point $z$:

$$
q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right) + z,\; 0,\; 2^b - 1\right),
\qquad \hat{x} = s\,(q - z)
$$

with $s = (\beta - \alpha)/(2^b - 1)$ for a chosen range $[\alpha, \beta]$. Symmetric
quantization fixes $z$ so that zero maps exactly to zero, which is what weight quantization
normally uses because it removes the cross-term in the integer matrix multiply and keeps the
accumulation cheap. Activations are usually asymmetric, since post-ReLU distributions are
one-sided.

**Where the error comes from.** Two sources, and they pull in opposite directions.

*Rounding.* For a fine grid and a smooth distribution, the rounding error is approximately
uniform on $[-s/2, s/2]$, so its variance is $s^2/12$. Widening the range to avoid clipping
increases $s$, so it increases this term quadratically.

*Clipping.* Anything outside $[\alpha, \beta]$ is saturated, and the error is unbounded for
a value far outside. Narrowing the range reduces $s$ and the rounding error, at the cost of
destroying outliers.

Calibration is the choice between the two. Min-max calibration never clips and therefore
lets one outlier activation inflate $s$ for the whole tensor. Percentile calibration
(clipping at, say, the 99.99th percentile) and MSE-optimal range search both accept some
clipping to buy resolution, and both usually beat min-max on activations. The reason this is
a real decision and not a hyperparameter sweep: the outliers in a detector's activations
may be the strong returns you care about.

**Per-tensor against per-channel.** A per-tensor weight scale means one $s$ for the whole
weight tensor, so a channel whose weights span a tenth of the tensor's range gets a tenth of
the available levels. Standard convolutions average over input channels and tolerate this.
Depthwise convolutions do not: each channel is its own filter with its own range, and
per-tensor INT8 PTQ on a depthwise network is the canonical failure case in the literature.
Per-channel (per-output-channel) weight scales fix it, at the cost of a vector of scales
that the accelerator has to support in its requantization path. Krishnamoorthi's whitepaper
reports per-channel weight plus per-layer activation quantization landing within about 2% of
float across a range of convolutional networks
([arXiv:1806.08342](https://arxiv.org/abs/1806.08342)); Nagel et al. give the modern
treatment of both PTQ and QAT pipelines
([arXiv:2106.08295](https://arxiv.org/abs/2106.08295)).

**PTQ against QAT.**

| | PTQ | QAT |
|---|---|---|
| Needs | a few hundred unlabelled representative inputs | the training pipeline, labels, a fine-tuning run |
| Cost | minutes to hours | a training run, plus the risk of drift from the original recipe |
| Mechanism | fit ranges to observed activations, fold batch norm, requantize | simulate quantization in the forward pass, backprop through it with a straight-through estimator |
| Use when | the drop is acceptable, or as the diagnostic that tells you whether QAT is needed | PTQ loses more than you can pay, typically on depthwise or low-bit models |
| Failure mode | one layer dominates the error and you do not know which | you retrain into a different optimum and lose a property you were not measuring |

QAT's straight-through estimator is worth being able to state: the forward pass applies
fake quantization $\hat{x} = s\,(\operatorname{round}(x/s) + z - z)$, whose derivative is
zero almost everywhere, so the backward pass substitutes the identity inside the clipping
range and zero outside. Say that and the follow-up question usually stops.

**The calibration set.** A few hundred samples, chosen to cover the deployment distribution
instead of the training distribution: for radar that means the range bands, clutter
environments, weather and target types you expect in the field. No labels are needed, which
is what makes PTQ cheap. A calibration set drawn only from easy conditions produces ranges
that clip exactly the returns that matter.

**What to check after quantizing a detector**, in the order you check them:

1. *Per-layer signal-to-quantization-noise ratio*, or the cosine similarity between float and
   quantized activations layer by layer, on the calibration inputs. Quantization damage is
   rarely spread evenly; one layer usually dominates, and the fix is to keep that layer in
   higher precision instead of retraining the whole network.
2. *Batch-norm folding correctness*. Folding BN into the preceding convolution before
   quantization changes the weight ranges, sometimes drastically, and a folding bug looks
   exactly like a quantization problem.
3. *The score distribution and the operating point.* A detector's threshold was tuned in
   float. Quantization shifts the score distribution, so the same numeric threshold is a
   different point on the precision-recall curve. Re-tune it, and report the metric at a
   matched operating point instead of at a matched threshold.
4. *The regression head separately from the classification head.* Localisation error degrades
   differently from class confidence, and small or distant objects lose first.
5. *Per-slice metrics.* By range band, by target class, by clutter condition. An aggregate
   detection metric that moves by half a point can hide a class that lost ten.
6. *Concatenation and elementwise-add points.* Operations that combine tensors require
   compatible scales, and a naive implementation inserts requantization steps that cost both
   accuracy and latency.
7. *First and last layers.* Both are commonly kept at higher precision; check whether yours
   were, because it changes what your "INT8 model" claim means.

### 3.2 Depthwise-separable convolutions

Derive this; do not quote it. A standard convolution with kernel $K \times K$, $C_{in}$
input channels and $C_{out}$ output channels, applied to an $H \times W$ feature map with
stride 1 and appropriate padding, costs

$$
\text{MACs}_{\text{std}} = H \cdot W \cdot C_{in} \cdot C_{out} \cdot K^2,
\qquad
\text{params}_{\text{std}} = C_{in} \cdot C_{out} \cdot K^2 .
$$

The separable version factors it into a depthwise stage (one $K \times K$ filter per input
channel, no mixing across channels) and a pointwise stage (a $1 \times 1$ convolution that
does the mixing):

$$
\text{MACs}_{\text{dw}} = H \cdot W \cdot C_{in} \cdot K^2,
\qquad
\text{MACs}_{\text{pw}} = H \cdot W \cdot C_{in} \cdot C_{out} .
$$

Their ratio to the standard convolution is

$$
\boxed{\;\frac{\text{MACs}_{\text{dw}} + \text{MACs}_{\text{pw}}}{\text{MACs}_{\text{std}}}
= \frac{1}{C_{out}} + \frac{1}{K^2}\;}
$$

For $K = 3$ and $C_{out} = 256$ that is $0.0039 + 0.111 \approx 0.115$, about 8.7 times fewer
multiply-accumulates, and the parameter ratio is the same expression. For $K=3$ the $1/K^2$
term dominates as soon as $C_{out}$ is more than a few dozen, so the saving asymptotes at
about 9 times and no channel-count choice improves on that
([MobileNets, arXiv:1704.04861](https://arxiv.org/abs/1704.04861); the inverted-residual
refinement is [MobileNetV2, arXiv:1801.04381](https://arxiv.org/abs/1801.04381)).

**Why they suit edge accelerators, and the caveat that earns you the staff mark.** The
parameter reduction is what gets the model into on-chip memory, which matters more on an
embedded part than the FLOP reduction does, because off-chip memory traffic dominates both
latency and energy. The caveat: the depthwise stage has terrible arithmetic intensity. It
performs $K^2$ multiply-accumulates per input element, around 9 for a 3 by 3 kernel, against
the hundreds a standard convolution performs, so it is memory-bandwidth-bound on most
hardware. An 8.7 times MAC reduction routinely produces a two to four times wall-clock
speedup, and on accelerators tuned for dense $C_{in} \times C_{out}$ matrix shapes it can
produce almost none. The general rule to state: a FLOP count is a claim about arithmetic,
and latency is a claim about the memory system, so measure on the device. The roofline
version of this argument is in
[hardware, memory and roofline](../part14-systems/04-hardware-memory-roofline.md).

**The accuracy cost.** Factoring removes the ability to mix spatial and cross-channel
information in one operation, so a separable network of the same depth has less capacity per
layer. In practice architectures compensate with more layers, wider pointwise stages, or
inverted residuals with an expansion factor. For a radar front end with few input channels
the trade is different from the image case: with small $C_{in}$ the depthwise stage is cheap
in absolute terms and the pointwise stage dominates, so check where your MACs actually are
before assuming the textbook ratio applies. **[FILL: where were yours?]**

### 3.3 Radar as a modality

What radar gives you that a camera does not:

* **Range, directly.** Time of flight gives distance without triangulation, monocular depth
  estimation or a second sensor.
* **Radial velocity, directly.** The Doppler shift gives closing speed per resolution cell,
  in a single frame, without tracking across frames. For an object as small as a UAV, the
  velocity channel often separates target from clutter better than any spatial feature does.
* **Operation in conditions that defeat optics.** Darkness, dust, fog, smoke, rain, glare.
* **Micro-Doppler structure.** Rotating propellers modulate the return and produce sidebands
  around the body Doppler, which is how a multirotor is distinguished from a bird or a bag in
  the wind. For small-UAV work this is often the discriminating signal, and it is a temporal
  signature rather than a spatial one, which changes the shape of the model you build.

The limits, which you should volunteer before you are asked:

* **Angular resolution is aperture-limited.** The beamwidth scales as $\theta \approx
  \lambda / D$ for an aperture of size $D$, so at practical sizes the cross-range resolution
  at a few hundred metres is metres to tens of metres. A radar does not see shape the way a
  camera does; it sees a point with attributes.
* **Range resolution is bandwidth-limited**, $\Delta R = c / (2B)$. At $B = 500$ MHz,
  $\Delta R = 0.3$ m. Wanting finer range detail means wanting more bandwidth, which is a
  spectrum and hardware question rather than a model question.
* **Doppler resolution is time-limited**, roughly $\Delta v = \lambda / (2 T)$ for a coherent
  processing interval $T$. Finer velocity discrimination costs dwell time, and dwell time is
  latency. This is the trade an ML engineer inherits from the signal-processing side and
  should be able to name.
* **Radar cross-section is small and fluctuating** for small UAVs, and clutter, multipath and
  ground returns are the dominant nuisance. The false-alarm problem is the hard problem.

The framing that makes this useful in a general ML interview: radar and camera fail
independently, which is the entire argument for fusing them, and the fusion is easiest in a
metric space (range and velocity) instead of in pixel space. Part XI's sensor-fusion chapter
(`part11-perception-autonomy/03-sensor-fusion.md`) carries the general treatment, and the
same "what does each modality actually carry" reasoning is the one you used on StreetSmart in
[chapter 02](02-streetsmart-deep-dive.md). The system-level version, where fusion has to fit
a compute budget, is in
[perception system design](../part17-ml-system-design/05-perception-system-av.md).

## 4. Probes they will push on

Each of these has a defensible answer from the theory above even when your project details
are thin. Where a blank appears, fill it before the interview.

**"Why INT8 rather than FP16?"** FP16 usually costs almost no accuracy and buys about half
the memory traffic, so it is the safer first move. INT8 is the right choice when the target
hardware has integer units that are materially faster or more power-efficient than its float
units, or when the model has to fit in on-chip memory. **[FILL: which was true of your
board?]** If the answer is "our accelerator only ran INT8", say that; hardware constraints
are legitimate reasons and interviewers prefer them to invented ones.

**"Why did PTQ not suffice?"** **[FILL: the PTQ accuracy drop and the metric.]** The
structural reason to expect trouble: depthwise layers have per-channel weight ranges that
vary by orders of magnitude, so per-tensor PTQ on a depthwise backbone degrades sharply.
Two things to try before QAT, and you should say whether you did: per-channel weight scales,
and a better range estimator than min-max on activations.

**"Where did the quantization error actually come from?"** Per-layer SQNR against the float
model. **[FILL: which layer dominated, if you remember.]** The general answer: the first
layer (wide dynamic range on the input representation), the depthwise layers (per-channel
range spread), and any layer feeding a concatenation with a mismatched scale.

**"How did you pick the calibration set?"** Size in the hundreds, sampled to cover the
deployment conditions instead of the training distribution. **[FILL: how many samples, and
how you chose them.]** Say what you would check: that the resulting activation ranges do not
clip the strong returns you care about.

**"What did quantization do to your false-alarm rate?"** The question behind it is whether
you know that the operating point moved. Scores shift, so a threshold tuned in float is a
different point on the curve after quantization. Report matched operating points.
**[FILL: what you observed.]**

**"How much of the speedup came from the architecture and how much from the numerics?"** You
need the ablation: float standard convolutions, float separable, INT8 separable, all measured
on the device. **[FILL: did you measure all three? If not, say so and give the design.]**

**"Your FLOP count says 9 times. What did you actually measure?"** The arithmetic-intensity
answer from section 3.2. If your measured speedup was much smaller, that is the expected
result and explaining why is stronger than the number itself.

**"What is the power number?"** **[FILL: joules per inference or average watts, if measured,
and the instrument you measured with.]** If you did not measure it, say that latency was the
budget you were held to and power was managed by the platform team; do not produce a figure
you cannot source.

**"How do you distinguish a drone from a bird?"** Micro-Doppler from the rotors, which is a
temporal signature, so the model needs enough coherent processing interval to resolve it, and
that dwell time competes with latency. **[FILL: what your model consumed: range-Doppler maps,
a Doppler spectrogram, a sequence of frames?]**

**"Why not fuse with a camera?"** Because they fail independently, and a camera adds the
angular resolution radar lacks. The cost is another sensor, another calibration, another
failure mode, and a fusion stage in a system with a hard power budget. **[FILL: was a camera
available on the platform?]**

**"Would you do this with a transformer today?"** A useful, honest answer: attention over
range-Doppler cells is attractive because the discriminating structure is relational rather
than local, and there are published radar transformer results. On an embedded board the
binding question is the same one as before, which is whether the target accelerator runs the
operator efficiently at the batch size of one, and depthwise-separable convolutions remain
easier to make fast on fixed-function hardware. What I would do is prototype both and measure
on the device rather than in FLOPs.

**"What would you do differently?"** **[FILL: one technical and one process answer, each with
a cost.]** A technically defensible candidate if it matches your memory: start with
per-channel PTQ and a proper per-layer error analysis before spending the QAT run, because
the analysis would have told you whether QAT was needed and which layers to protect.

## 5. Blanks to fill before you present this

- [ ] What the model consumed: raw ADC, range-Doppler map, range-azimuth map, or CFAR detections.
- [ ] What decision the output drove, and what a false positive cost.
- [ ] The compute platform, its memory, and its power envelope.
- [ ] The latency or frame-rate budget, and where it came from.
- [ ] The baseline that existed before your work, and by what factor it missed the budget.
- [ ] Your role, the team, and who owned the signal-processing front end.
- [ ] Network depth, input resolution, and MAC and parameter counts before and after the
      separable rewrite.
- [ ] The PTQ accuracy drop, on which metric, and what made you move to QAT.
- [ ] Per-channel or per-tensor weights; which layers stayed in higher precision.
- [ ] Calibration set size and how it was sampled.
- [ ] Latency before and after, on the device, and how it was measured.
- [ ] Accuracy before and after, per range band or target class if available.
- [ ] Power or energy per inference, and the instrument, or an explicit "we did not measure it".
- [ ] Which layer dominated the quantization error.
- [ ] What quantization did to the operating point and the false-alarm rate.
- [ ] Whether you measured the float-separable ablation, separating architecture from numerics.
- [ ] Two rejected alternatives with their disqualifying clauses.
- [ ] The surprise, the diagnosis and the fix.
- [ ] One technical and one process thing you would do differently.

## Exercises

**★ 1. Derive the separable ratio.** From first principles, derive $\frac{1}{C_{out}} +
\frac{1}{K^2}$ and evaluate it for $K=3, C_{out}=64$ and for $K=5, C_{out}=512$.

??? success "Solution"
    $K=3, C_{out}=64$: $1/64 + 1/9 = 0.0156 + 0.1111 = 0.127$, about 7.9 times fewer MACs.
    $K=5, C_{out}=512$: $1/512 + 1/25 = 0.002 + 0.04 = 0.042$, about 24 times. Larger kernels
    benefit more, which is why separable blocks with 5 by 5 or 7 by 7 depthwise kernels show
    up in efficient architectures: the spatial extent is nearly free once the channel mixing
    has been factored out.

**★★ 2. Quantization error budget.** A weight tensor has range $[-0.8, 0.8]$ per tensor, but
one output channel spans $[-0.02, 0.02]$. With symmetric INT8, how many distinct levels does
that channel receive under per-tensor and under per-channel scaling?

??? success "Solution"
    Per-tensor symmetric INT8 uses $s = 0.8/127 \approx 0.0063$. The narrow channel spans
    $0.04$, so it occupies about $0.04/0.0063 \approx 6.3$ levels out of 255: roughly 2.7
    bits of the 8. Per-channel scaling gives it $s = 0.02/127$ and the full 255 levels. This
    is the depthwise failure case in one calculation, and it is worth having at your
    fingertips, because it answers "why per-channel?" with arithmetic instead of an appeal to
    a paper.

**★★ 3. Radar resolution arithmetic.** Your radar has 300 MHz of bandwidth and operates at
24 GHz with a coherent processing interval of 20 ms. Give range resolution and velocity
resolution, and say what each one implies for the model.

??? success "Solution"
    Range: $\Delta R = c/(2B) = 3 \times 10^8 / (6 \times 10^8) = 0.5$ m. Velocity:
    $\lambda = c/f = 0.0125$ m, so $\Delta v = \lambda/(2T) = 0.0125/0.04 \approx 0.31$ m/s.
    Implications: two targets closer than half a metre in range share a cell, so the model
    cannot separate them from a single frame and must rely on Doppler or on tracking; and the
    velocity grid is fine enough that micro-Doppler sidebands from rotor blades are
    resolvable, which is what makes UAV classification feasible at all. Longer dwell buys
    finer velocity resolution and costs latency, which is the trade to name out loud.

**★★★ 4. Design the ablation you wish you had run.** Write the experiment matrix that
separates the architecture change from the numerics change, with the metrics and the
measurement conditions.

??? success "Solution"
    Four cells: {standard, separable} by {FP32, INT8}, each measured for accuracy on a fixed
    held-out set with per-range-band and per-class breakdowns, and for latency on the target
    device at batch size 1 with the same pre and post-processing, reporting mean and P95 over
    a few hundred runs after warm-up. Add two diagnostics: per-layer SQNR for the INT8 cells,
    and a matched-operating-point comparison instead of a matched-threshold one. The matrix
    answers the question an interviewer will ask ("which change bought the speed?") and
    exposes an interaction, since separable networks lose more to per-tensor INT8 than
    standard ones do.

## References

* Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference: a
  whitepaper*, [arXiv:1806.08342](https://arxiv.org/abs/1806.08342). Per-channel weights,
  per-layer activations, and the depthwise failure case.
* Nagel et al., *A White Paper on Neural Network Quantization*,
  [arXiv:2106.08295](https://arxiv.org/abs/2106.08295). The modern PTQ and QAT pipelines.
* Jacob et al., *Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference*, [arXiv:1712.05877](https://arxiv.org/abs/1712.05877).
  The integer-only scheme and simulated quantization during training.
* Howard et al., *MobileNets*, [arXiv:1704.04861](https://arxiv.org/abs/1704.04861); Sandler
  et al., *MobileNetV2*, [arXiv:1801.04381](https://arxiv.org/abs/1801.04381).
* This book: [quantization](../part06-llm-training/05-quantization.md),
  [inference systems](../part14-systems/03-inference-systems.md),
  [hardware, memory and roofline](../part14-systems/04-hardware-memory-roofline.md),
  [CNN architectures](../part04-vision/03-cnn-architectures.md).
