# Zoox, Nuro, Aurora & robotaxi peers

> **Why this matters at staff level.** These three companies made different bets on
> the *product* around the driver: Zoox built a bidirectional vehicle with no
> steering wheel and its own symmetric sensor rig; Nuro pivoted from delivery robots
> to licensing an "AI-first" driver to carmakers and ride-hail platforms; Aurora put
> a long-range FMCW lidar on Class 8 trucks and argued its safety in a public
> safety case. Their interviews test whether you can reason from the product
> constraint (vehicle geometry, who owns the fleet, highway speed) to the perception,
> prediction and safety design — and whether you distinguish what they have published
> from what you are guessing.

## TL;DR — the interview card

- **Zoox** (Amazon): purpose-built bidirectional robotaxi; four symmetric corner
  sensor pods (cameras, lidar, radar, long-wave infrared, microphones) with
  overlapping fields of view; ML perception plus an independent geometric
  collision-avoidance layer described in its safety materials; TeleGuidance as
  guidance not control; public service in Las Vegas (Sept 2025 journal post).
  Its journal has technical posts on perception and on **sensor staleness**.
- **Nuro**: founded 2016 by ex-Google self-driving engineers; R1–R3 delivery vehicles;
  now licenses the **Nuro Driver** (L4, "AI-first") to partners; public engineering
  posts on **CIMRL** (combining imitation and RL for safe driving, arXiv:2406.08878),
  a **unified perception model**, and an in-house model-compiler framework.
- **Aurora**: the Aurora Driver for long-haul trucking; commercial driverless
  freight in Texas from May 2025, night operations by August 2025; **FirstLight**
  FMCW lidar (per-point velocity, 450 m range per the company); **Verifiable AI**
  (June 2024) and the public **Safety Case Framework** (proficient, fail-safe,
  continuously improving, resilient, trustworthy).
- The shared interview themes: sensor-rig design from vehicle geometry, prediction
  and planning that mix imitation with explicit constraints, remote assistance that
  stays out of the control loop, and safety arguments structured as claims and
  evidence.

## 1. The business in one paragraph (each)

**Zoox**, an Amazon subsidiary, builds a purpose-built, bidirectional, four-seat
robotaxi with no manual controls and operates it as a ride-hailing service; the
company tested on retrofitted vehicles for years and opened public rides in Las Vegas
in September 2025. Owning the vehicle design lets Zoox place sensors where the
perception problem wants them, and the bidirectional body removes the need to turn
around, but every component (including the safety argument for a vehicle with no
steering wheel) has to be built and justified in-house.

**Nuro** began as an autonomous delivery company (the R1, R2 and R3 vehicles, with
retail partners) and has repositioned to license the Nuro Driver, an L4 system it
describes as AI-first, together with a toolkit, to vehicle makers and ride-hail
platforms; its public partners include Uber and Lucid for a robotaxi programme.
The business consequence is that the driver must be vehicle-agnostic and
deployable on partner compute, which is why Nuro publishes about model compilation
and unified models.

**Aurora** builds the Aurora Driver for freight, integrated into Volvo and PACCAR
trucks with hardware partners including Continental and NVIDIA, and launched
commercial driverless trucking between Dallas and Houston in May 2025, extending to
night operations in August 2025. Highway trucking changes the perception problem
(long range at speed, fewer vulnerable road users, more weather and night) and the
safety argument (a public safety case framework and a stated "verifiable AI"
philosophy).

## The ML problems that define the companies

| Problem | Why it is hard | Public evidence |
|---|---|---|
| **Sensor rig design from vehicle geometry** (Zoox) | A symmetric bidirectional vehicle needs perception that is identical in both directions and at every corner; thermal and audio add modalities with different physics. | Zoox journal "Perception"; Zoox vehicle announcements. |
| **Sensor timing and staleness** (Zoox) | Sensors arrive at different rates and with jitter; a model that assumes fresh inputs fails silently when one stream lags. | Zoox journal post on sensor staleness. |
| **Remote assistance that is not remote driving** (Zoox, Nuro) | Humans must help with ambiguous scenes without adding network latency to the safety case. | Zoox TeleGuidance (public materials); Nuro remote operations for delivery. |
| **A vehicle-agnostic learned driver** (Nuro) | Licensing means the stack must port across sensor sets and compute; models must be compiled for partner hardware. | Nuro blog: unified perception model; model-compiler framework. |
| **Learning to drive safely, not just human-like** (Nuro, Aurora) | Pure imitation compounds error and copies human mistakes; pure RL needs reward engineering and a realistic simulator. | Nuro CIMRL (arXiv:2406.08878); Aurora, "AI Alignment: Ensuring the Aurora Driver is Safe and Human-Like". |
| **Long-range perception at highway speed** (Aurora) | Stopping distance for a loaded truck requires detection hundreds of metres out, at night and in weather. | Aurora FirstLight FMCW lidar posts; company site (450 m, night pedestrian detection). |
| **A safety argument for a learned system** (Aurora) | Learned components must be wrapped in a structure whose behaviour can be verified against requirements. | Aurora "Verifiable AI" (2024); Safety Case Framework (2020, public site). |

## 2. The stacks as publicly described

```mermaid
flowchart TB
  subgraph Z["Zoox (public)"]
    Z1[4 corner sensor pods:<br/>cameras, lidar, radar,<br/>LWIR thermal, microphones] --> Z2[ML perception<br/>timestamp-aware, trained<br/>with synthetic staleness]
    Z2 --> Z3[Prediction + planning<br/><i>details not public</i>]:::inf
    Z1 --> Z4[Independent geometric<br/>collision avoidance<br/><i>safety materials</i>]
    Z3 --> Z5[Bidirectional vehicle,<br/>4-wheel steering]
    Z4 --> Z5
    Z3 -.-> Z6[TeleGuidance:<br/>guidance, not control]
  end
  subgraph N["Nuro (public)"]
    N1[Partner vehicle sensors] --> N2[Unified perception model<br/><i>blog</i>]
    N2 --> N3[Planner: imitation priors +<br/>RL selection with safety<br/>constraints (CIMRL)]
    N3 --> N4[Deployment via in-house<br/>model compiler (FTL)<br/><i>blog</i>]
    N4 --> N5[Licensed 'Nuro Driver'<br/>on partner platforms]
  end
  subgraph A["Aurora (public)"]
    A1[FirstLight FMCW lidar<br/>+ cameras + radar] --> A2[Perception<br/>long-range, per-point velocity]
    A2 --> A3[Learned prediction/planning<br/>inside a verifiable structure<br/><i>'Verifiable AI' post</i>]
    A3 --> A4[Aurora Driver on<br/>Volvo / PACCAR trucks;<br/>NVIDIA + Continental hardware]
    A3 --> A5[Safety Case Framework:<br/>claims + evidence]
  end
  classDef inf fill:#fff3cd,stroke:#b8860b;
```

**Known versus inferred.** Sensor modalities, the existence of TeleGuidance, the
staleness post, the CIMRL paper, the compiler and unified-model posts, the FMCW lidar
and the safety-case structure are public. Everything about the internal prediction
and planning architectures at Zoox and Aurora is inferred from the posts' framing,
and is shaded or marked accordingly.

## 3. Deep dives

### 3.1 Zoox: the vehicle *is* the sensor rig

**The problem.** A retrofitted car inherits a forward-facing sensor layout and a
driver's blind spots. A bidirectional vehicle that drives equally well both ways,
with no driver, needs perception that is symmetric, redundant at every corner, and
able to see around the vehicle's own body in tight urban spaces.

**The approach (public).** Zoox places a sensor pod at each of the four corners; each
pod carries cameras, lidar and radar, with long-wave infrared (thermal) cameras and
microphones (for sirens) in the suite, so that each corner has a wide field of view
and the pods overlap around the vehicle. Because the vehicle is symmetric, one
perception model serves both driving directions; because the pods overlap, an
occluded or failed sensor at one corner is covered by its neighbours. The company's
safety materials also describe an independent collision-avoidance capability that
uses geometric methods rather than learned components, as a layer beneath the ML
stack. The fusion mathematics is [Sensor fusion](../part11-perception-autonomy/03-sensor-fusion.md);
thermal imaging is a different physics (emitted radiation, no dependence on
illumination) and is best treated as an additional camera modality in a
feature-level fusion.

**The trade-off.** Designing the vehicle around perception buys symmetric coverage
and redundancy, at the cost of building and certifying a vehicle with no manual
controls, a much longer road to public service than a retrofit, and a fleet that
cannot be scaled by buying cars. The rejected alternative, a retrofit of a
production car, is what every other robotaxi company chose first; Zoox tested on
retrofits too, but shipped the purpose-built vehicle.

**Sources.** Zoox journal, "Perception"; Zoox journal, Las Vegas public launch (Sept
2025); Zoox testing announcements (Seattle 2021, Austin and Miami 2024).

!!! tip "How to say it in the interview"
    "If I own the vehicle design, I would design the sensor rig for the perception
    problem rather than the other way round: symmetric corner pods with overlapping
    fields of view so that one model serves both driving directions and every corner
    is covered twice, which is the layout Zoox describes for its bidirectional
    robotaxi. I would fuse at the feature level and treat thermal as an extra camera
    modality with different physics, useful at night and for pedestrians. I would
    keep an independent geometric collision-avoidance layer beneath the learned
    stack, as Zoox's safety materials describe, so the safety argument does not rest
    entirely on a neural network. The alternative I would reject is a retrofit
    layout with a forward bias; it is faster to market but inherits the driver's blind
    spots. The trade-off is the time and cost of certifying a vehicle with no manual
    controls. I would evaluate perception with per-corner ablations, because
    redundancy is only real if the metrics survive losing a pod, and with a
    night-and-thermal slice."

### 3.2 Zoox: sensor staleness as a modelling problem

**The problem.** Multi-sensor perception assumes inputs that arrive together and on
time. In practice streams lag, drop and jitter; a model trained only on well-aligned
data behaves unpredictably on stale inputs, and the failure is silent.

**The approach (public).** Zoox's journal post on sensor staleness describes making
the model aware of timing, feeding timestamp information as features, and training
with synthetically staled inputs so the network learns to weight recent evidence and
to degrade gracefully when a stream is late. This is a robustness-by-augmentation
pattern: the training distribution is widened to include the failure mode, and the
model is given the information it needs to detect it. Compare the temporal-alignment
mechanisms in [Occupancy & temporal perception](../part11-perception-autonomy/05-occupancy-temporal.md)
and the reliability material in
[Uncertainty & reliability](../part13-retrieval-eval-reliability/03-uncertainty-reliability.md).

**The trade-off.** Timestamp-aware training adds inputs and augmentation complexity
and can slightly reduce accuracy on perfectly aligned data; it buys predictable
behaviour under the timing faults a real fleet sees. The rejected alternative is a
hard gate that drops any frame set with a stale stream, which sacrifices
availability and still leaves the boundary case.

**Source.** Zoox journal post on sensor staleness (details as published in the post).

!!! tip "How to say it in the interview"
    "I would make timing a first-class model input and a first-class augmentation,
    which is the approach Zoox describes in its sensor-staleness post: give the
    network per-sensor timestamps and train with synthetically delayed streams so it
    learns to discount stale evidence instead of trusting it. The alternative I would
    reject is a hard freshness gate; it trades availability for a boundary that still
    fails at the threshold. The trade-off is a small accuracy cost on clean data and
    more complex data loaders. I would evaluate with a staleness sweep, accuracy as a
    function of injected delay per modality, and I would monitor real per-sensor
    latency in production so the augmentation matches the fleet's actual
    distribution."

### 3.3 Zoox and Nuro: remote assistance that stays out of the control loop

**The problem.** Ambiguous scenes (a construction zone with hand signals, an
unexpected closure) will exceed the driver's competence; a human must help. If that
help is remote driving, network latency and human reaction time enter the safety
case; if it is absent, the vehicle strands.

**The approach (public).** Zoox describes TeleGuidance as remote humans providing
guidance, such as suggested paths or interpretations, while the vehicle keeps
responsibility for safe execution; Nuro operated remote operations for its delivery
fleet on the same principle; Waymo's Fleet Response post (see the
[Waymo chapter](waymo.md#36-operating-a-rider-only-fleet-fleet-response)) draws the
same line. The ML content is the *request* policy: a calibrated uncertainty signal
decides when to ask, and every request is a labelled hard case for the data engine.

!!! tip "How to say it in the interview"
    "I would keep humans as advisors, not drivers, which is the design Zoox calls
    TeleGuidance and Waymo calls Fleet Response: the vehicle asks for a suggested
    path or interpretation and remains responsible for executing it safely. I would
    reject teleoperation because it puts network latency and human reaction time
    inside the safety case. The request policy would be a calibrated uncertainty
    threshold with a time budget, and every request would become a training example.
    I would measure requests per thousand miles, time-to-resolution, and the fraction
    that stop being requests after the next model release."

### 3.4 Nuro: a licensable, AI-first driver — CIMRL, unified perception, and a compiler

**The problem.** Licensing a driver means it must generalise across partner vehicles
and compute, and it must be safe in closed loop, not merely human-like in open loop.
Pure behaviour cloning copies human errors and compounds its own; pure RL needs a
reward that captures driving and a simulator realistic enough to train in.

**The approach (public).** Nuro's CIMRL paper ("Combining Imitation and
Reinforcement Learning for Safe Autonomous Driving", 2024) proposes a framework in
which imitation-learned motion priors provide candidate behaviours and reinforcement
learning, with explicit safety constraints, learns to choose among and improve them
in simulation, reducing the reward engineering that pure RL needs while avoiding the
compounding error of pure imitation. The blog describes the same idea for a general
audience. Nuro's unified-perception-model post describes consolidating perception
tasks into one multi-task model (a single backbone with many heads, the same
economics as Tesla's HydraNet) and its model-compiler post describes an in-house
framework for compiling models to run on vehicle hardware, the necessary condition
for deploying on partners' compute. The RL and imitation mathematics are in
[Imitation learning](../part12-rl/05-imitation-learning.md) and
[Policy gradients & PPO](../part12-rl/04-policy-gradients-ppo.md).

**The trade-off.** Imitation-plus-RL buys closed-loop safety improvements without
hand-written rewards, at the cost of dependence on simulator fidelity and on the
imitation prior's coverage. The rejected alternatives are named in the paper's
framing: pure imitation (compounding error, copying mistakes) and pure RL (reward
design, sample cost).

**Sources.** Nuro blog, "CIMRL: Combining Imitation and Reinforcement Learning for
Safe Autonomous Driving" and the paper (arXiv:2406.08878); Nuro blog, "Unified
Perception Model"; Nuro blog, "FTL Model Compiler Framework".

!!! tip "How to say it in the interview"
    "For a planner that has to be safe in closed loop, I would start from an
    imitation prior and add reinforcement learning with explicit safety constraints
    on top, letting RL select and refine among imitation-proposed behaviours, which
    is the structure of Nuro's CIMRL paper (2024). I would reject pure behaviour
    cloning because it compounds error and copies human mistakes, and pure RL because
    reward engineering for driving is where projects die. The trade-off is
    dependence on the simulator and on the prior's coverage, so I would validate the
    simulator against logged outcomes and keep the imitation data broad. For a
    licensable driver I would also consolidate perception into one multi-task model
    and invest in a compiler path to partner hardware, as Nuro's unified-perception
    and model-compiler posts describe, because portability is the product. I would
    evaluate with closed-loop safety metrics, collisions and constraint violations,
    on scenario slices, alongside human-likeness metrics, and I would report both
    because they trade off."

### 3.5 Aurora: long-range FMCW lidar for trucks

**The problem.** A loaded truck at highway speed needs several hundred metres of
detection range to stop, at night and in weather, and it needs to know not just
where a distant object is but whether it is moving.

**The approach (public).** Aurora's FirstLight lidar is frequency-modulated
continuous-wave: it measures per-point radial velocity via the Doppler shift in
addition to range, and the company states it can see over 450 metres and detect a
pedestrian at night seconds earlier than a human at highway speed. Per-point velocity
turns a static point cloud into a partially dynamic one, which simplifies detection
of moving objects at range and reduces dependence on temporal accumulation. Aurora
describes the technology in newsroom posts ("FMCW lidar: the self-driving
game-changer"; "FirstLight lidar on a chip"). Detection and fusion math:
[3D perception](../part04-vision/07-3d-perception.md), [Sensor fusion](../part11-perception-autonomy/03-sensor-fusion.md).

**The trade-off.** FMCW gives velocity and long range and is less susceptible to
interference than time-of-flight, at the cost of a proprietary sensor programme
(Aurora acquired the technology and built it in-house) and lower point density per
frame; the rejected alternative is off-the-shelf time-of-flight lidar with velocity
inferred from tracking, which is adequate for city speeds but late at highway range.

!!! tip "How to say it in the interview"
    "For highway trucking I would prioritise range and per-point velocity over point
    density, which is why Aurora's FirstLight FMCW lidar is the reference point: it
    returns Doppler velocity for every point and the company states a range beyond
    450 metres. I would use the velocity channel directly in the detector so moving
    objects at range are separable from clutter without waiting for several frames.
    The alternative I would reject is time-of-flight lidar plus tracking-derived
    velocity; at truck stopping distances the extra frames cost time. The trade-off
    is a proprietary sensor programme and sparser returns, so I would fuse with radar
    and long-focal cameras for classification at range. I would evaluate detection
    recall as a function of range and time-to-collision, at night and in rain, which
    is the metric that matters for a truck."

### 3.6 Aurora: Verifiable AI and the Safety Case Framework

**The problem.** The most human-like driver comes from learning; the most auditable
driver comes from rules. A freight operator, a regulator and an insurer need an
argument that the learned system is acceptably safe, structured so that evidence
can be attached to each claim.

**The approach (public).** Aurora published its Safety Case Framework in 2020, a
structured argument that the Aurora Driver is acceptably safe to operate on public
roads, decomposed into top-level claims (proficient, fail-safe, continuously
improving, resilient, trustworthy) and sub-claims, hosted publicly. In June 2024
Chris Urmson's "Aurora's Verifiable AI Approach to Self-Driving" describes using
machine learning extensively while keeping the system in a structure whose behaviour
can be verified against requirements, rather than a single opaque end-to-end model;
Drew Bagnell's "AI Alignment: Ensuring the Aurora Driver is Safe and Human-Like"
and "AI Transparency: The Why and How" (2024) elaborate learning from human driving
within explicit safety bounds and making decisions inspectable. The company reported
completing its safety case for the launch lanes ahead of the May 2025 driverless
start. The book's treatment of safety arguments and failure modes is in
[Safety & failure modes](../part15-interpretability-safety/02-safety-failure-modes.md).

**The trade-off.** A verifiable structure constrains architecture (interfaces must
exist for evidence to attach to) and may cap human-likeness relative to an
end-to-end policy; it buys a safety argument that can be audited and a path to
insurance and regulation. The rejected alternative is the end-to-end black box the
[Tesla chapter](tesla.md#34-lanes-as-language-planning-as-search-and-the-move-to-end-to-end)
describes; Aurora's posts argue against it for driverless freight.

**Sources.** Aurora, "Aurora Unveils First-Ever Safety Case Framework" (2020) and the
framework site; Urmson, "Aurora's Verifiable AI Approach to Self-Driving" (2024);
Bagnell, "AI Alignment: Ensuring the Aurora Driver is Safe and Human-Like" (2024) and
"AI Transparency: The Why and How" (2024); Aurora newsroom on driverless launch and
night operations (2025).

!!! tip "How to say it in the interview"
    "I would structure the driver so that a safety case can attach evidence to it:
    learned perception and prediction, learned proposals in planning, but explicit
    interfaces and constraints where requirements must be verified, which is what
    Aurora's Verifiable AI post (2024) argues and what its public Safety Case
    Framework, with claims like proficient, fail-safe and resilient, is built to
    consume. I would reject a single end-to-end policy for driverless freight, not
    because it cannot drive well, but because I could not attach evidence to its
    decisions for a regulator or insurer. The trade-off is a ceiling on
    human-likeness and slower iteration at the interfaces, which Aurora's alignment
    post addresses by learning from human driving inside the bounds. I would evaluate
    each safety-case claim with its own evidence stream: scenario-sliced closed-loop
    metrics for proficiency, fault-injection tests for fail-safe, and release-over-release
    trend lines for continuous improvement."

## 4. Likely interview questions

!!! interview "Q1. (Zoox) Design the perception stack for a symmetric bidirectional vehicle with corner pods. How does symmetry change the model and the data?"
    **Answer sketch.** One model serves both directions by canonicalising the vehicle
    frame; augment by mirroring and by rotating the pod assignment; per-pod ablations
    as a regression gate; feature-level fusion across pods with overlapping fields of
    view; thermal and audio as extra modalities. Data: label once in the vehicle
    frame; verify pod calibration continuously. Links: [Sensor fusion](../part11-perception-autonomy/03-sensor-fusion.md), [AV perception design](../part17-ml-system-design/05-perception-system-av.md).

    !!! tip "How to say it in the interview"
        "I would canonicalise everything into the vehicle frame so one model serves
        both directions, and I would use the symmetry as augmentation, mirroring and
        rotating pod assignments, which is the advantage Zoox's four-corner layout
        gives. I would reject direction-specific models; they halve the data. The
        trade-off is that a calibration error at one pod now contaminates both
        directions, so I would gate on per-pod ablations and monitor calibration
        continuously."

!!! interview "Q2. (Zoox) One lidar stream is arriving 150 ms late intermittently. What happens to your detector and what do you change?"
    **Answer sketch.** Without timing awareness the fusion silently merges stale
    geometry with fresh images, producing ghosts and misplaced boxes. Fix: timestamp
    features, synthetic staleness augmentation, per-sensor health inputs, and a
    staleness sweep in evaluation; production latency monitoring. Link:
    [Occupancy & temporal perception](../part11-perception-autonomy/05-occupancy-temporal.md).

    !!! tip "How to say it in the interview"
        "The failure is silent misalignment, ghosts and misplaced boxes, so I would
        do what Zoox's staleness post describes: feed timestamps, train with
        synthetic delay, and evaluate with a delay sweep. I would reject dropping the
        frame set; availability matters. The trade-off is a small clean-data cost."

!!! interview "Q3. (Zoox / Nuro / Waymo) Design remote assistance for a driverless fleet. What is the interface, and what must it never do?"
    **Answer sketch.** Guidance not control; calibrated request policy; time budget
    and safe fallback while waiting; UI that presents interpretations and candidate
    paths; every request logged as training data; metrics as above.

    !!! tip "How to say it in the interview"
        "It must never drive. Zoox's TeleGuidance and Waymo's Fleet Response both
        keep the vehicle responsible and the human advisory, and I would follow that
        line so network latency stays out of the safety case. I would evaluate
        requests per mile and resolution time, and treat every request as a labelled
        hard case."

!!! interview "Q4. (Nuro) Your imitation-learned planner is human-like but has a 2× collision rate in closed-loop sim. What do you do?"
    **Answer sketch.** Diagnose compounding error versus copied human mistakes;
    add RL on top of the imitation prior with safety constraints (CIMRL); add
    on-policy corrections; check simulator realism before trusting the number;
    report safety and human-likeness separately. Link: [Imitation learning](../part12-rl/05-imitation-learning.md).

    !!! tip "How to say it in the interview"
        "I would keep the imitation prior and add constrained RL that selects and
        refines among its proposals in simulation, which is Nuro's CIMRL design,
        because it targets exactly the closed-loop gap without hand-written rewards.
        I would first verify the simulator is not the culprit. The trade-off is
        simulator dependence, so I would validate against logs, and I would report
        safety and human-likeness as separate metrics."

!!! interview "Q5. (Nuro) You license the driver to a partner with a different sensor set and compute. What has to be true of your stack?"
    **Answer sketch.** Sensor-abstracted perception interfaces; a unified multi-task
    model retrainable per rig; calibration tooling; a compiler path to the partner's
    accelerators with quantization and operator coverage; a portable evaluation
    suite. Link: [ML platform](../part17-ml-system-design/12-ml-platform-feature-store-monitoring.md).

    !!! tip "How to say it in the interview"
        "Portability is the product, so I would consolidate perception into one
        multi-task model that is retrained per sensor rig behind a stable interface,
        and I would own a compiler path to partner hardware, which is what Nuro's
        unified-perception and model-compiler posts describe. I would reject
        per-partner forks of the stack. I would evaluate with the same scenario suite
        on every platform and gate on parity."

!!! interview "Q6. (Aurora) Why does per-point velocity from FMCW lidar matter for a truck, and what does it not solve?"
    **Answer sketch.** It separates moving objects at range immediately, reduces
    reliance on multi-frame accumulation, and improves tracking initialisation; it
    does not give classification at range (need cameras), does not help with
    lateral velocity (radial only), and returns fewer points per frame.

    !!! tip "How to say it in the interview"
        "Doppler velocity lets me separate a moving object at 400 metres in one
        frame, which is the stopping-distance argument behind Aurora's FirstLight;
        it does not classify, and it only sees radial velocity, so I would fuse
        long-focal cameras for class and use tracking for lateral motion. I would
        evaluate recall by range and time-to-collision."

!!! interview "Q7. (Aurora) Write the top level of a safety case for a driverless truck lane and say what evidence each claim needs."
    **Answer sketch.** Claim: acceptably safe on the lane. Sub-claims (Aurora's
    framework): proficient (closed-loop scenario metrics, on-road data), fail-safe
    (fault injection, minimal-risk-condition tests), continuously improving
    (release trend lines, incident learning), resilient (cyber, adverse conditions),
    trustworthy (operations, transparency). Evidence must be measurable and
    versioned. Link: [Safety & failure modes](../part15-interpretability-safety/02-safety-failure-modes.md).

    !!! tip "How to say it in the interview"
        "I would decompose the safety claim the way Aurora's public Safety Case
        Framework does, proficient, fail-safe, continuously improving, resilient and
        trustworthy, and attach a measurable evidence stream to each: scenario-sliced
        closed-loop results, fault-injection tests, release trends, adverse-condition
        tests and operational audits. I would reject a single aggregate safety metric
        because it cannot support the argument's structure."

!!! interview "Q8. (Aurora) 'Verifiable AI' versus end-to-end: where would you draw the line in the architecture?"
    **Answer sketch.** Learned perception and prediction with measurable interfaces;
    learned proposals in planning; explicit constraint checking and a rule-based
    minimal-risk fallback outside the learned policy; logged rationales for
    transparency (Bagnell's transparency post). Name what is lost.

    !!! tip "How to say it in the interview"
        "I would let learning own perception, prediction and trajectory proposals,
        and keep constraint enforcement and the minimal-risk fallback outside the
        learned policy, which is how I read Aurora's Verifiable AI post and its
        transparency post. I would reject a monolithic end-to-end driver for freight
        because I could not attach evidence to it. The trade-off is a ceiling on
        human-likeness, which the alignment post addresses by learning from human
        driving within bounds."

!!! interview "Q9. (All) Night driving: how do you validate perception for night operations before launching them?"
    **Answer sketch.** Night-sliced datasets; thermal (Zoox) or FMCW range
    (Aurora) as the modality that carries night; illumination-conditioned metrics;
    closed-loop night scenarios; a staged launch (Aurora went driverless in the day
    first, then at night in August 2025).

    !!! tip "How to say it in the interview"
        "I would launch night operations as a separate release with its own
        evidence, night-sliced perception metrics and closed-loop night scenarios,
        which is the staging Aurora followed, going driverless by day in May 2025
        and at night in August 2025. I would lean on the modality that carries
        night, thermal or long-range lidar, and evaluate recall by range at low
        illumination."

!!! interview "Q10. (All) Coding: implement a 2D Kalman filter for a tracked object with a radial-velocity measurement."
    **Answer sketch.** State $[x, y, v_x, v_y]$, constant-velocity transition,
    measurement of position plus radial velocity $v_r = (x v_x + y v_y)/\sqrt{x^2+y^2}$,
    which is non-linear, so an extended Kalman update with the Jacobian; shape
    comments; test on a synthetic straight-line track. Reference:
    [Tracking](../part11-perception-autonomy/04-tracking.md).

    !!! tip "How to say it in the interview"
        "Radial velocity is non-linear in the state, so I would write an extended
        Kalman update with the Jacobian of the measurement, and test on a synthetic
        track where the true velocity is known. That is the measurement an FMCW lidar
        like Aurora's provides per point, and the reason it helps tracking
        initialisation."

!!! interview "Q11. (All) How would you compare the three companies' bets in one minute, and which would you copy for a new urban robotaxi?"
    **Answer sketch.** Zoox: own the vehicle for perception and product; Nuro: be
    vehicle-agnostic and license; Aurora: pick the domain (highway freight) where
    range and a safety case are decisive. For urban robotaxi: Waymo/Zoox-style
    redundant sensing plus a Nuro-style imitation-plus-RL planner and an explicit
    safety structure.

    !!! tip "How to say it in the interview"
        "Zoox bet on the vehicle, Nuro on portability, Aurora on a domain where
        range and a public safety case decide. For an urban robotaxi I would copy
        redundant symmetric sensing from Zoox, the imitation-plus-constrained-RL
        planner from Nuro's CIMRL, and the claims-and-evidence safety structure from
        Aurora, and I would say plainly which parts are my inference."

!!! interview "Q12. (All) Design the evaluation that decides a new city is ready."
    **Answer sketch.** Map and ODD definition; scenario bank mined from local
    driving; closed-loop sim on that bank; supervised miles with a safety driver;
    matched human benchmarks for the city; a release checklist tied to safety-case
    claims. Link: [Evaluation](../part13-retrieval-eval-reliability/02-evaluation.md).

    !!! tip "How to say it in the interview"
        "I would define the operational domain, mine a city-specific scenario bank,
        run closed-loop simulation on it, then supervised miles, and I would gate on
        safety-case claims with city-matched human benchmarks, the way Waymo's safety
        hub reports by geography. I would reject reusing another city's numbers as
        evidence."

!!! interview "Q13. (All) What is the single most common failure in perception teams you have seen, and how do these companies' write-ups address it?"
    **Answer sketch.** Silent distribution shift (timing, calibration, weather,
    new geography) that average metrics hide. Zoox's staleness training, Aurora's
    range-and-night claims, and scenario-sliced evaluation everywhere are the
    published answers.

    !!! tip "How to say it in the interview"
        "Silent shift hidden by averages. I would answer it the way the public posts
        do: widen the training distribution to include the failure, as Zoox does for
        staleness, define the operating envelope explicitly, as Aurora does with
        range and night, and gate on slices rather than means."

## 5. What to bring from your background

* **Detection and tracking at scale**: all three interview on 3D detection,
  multi-sensor association and tracking; bring concrete numbers on how you improved
  recall on a hard slice and how you measured it.
* **Robustness engineering** (augmentation for failure modes, calibration
  monitoring, latency budgets) is directly relevant to Zoox's staleness work and to
  Aurora's night and weather claims.
* **Imitation and RL literacy**: Nuro's CIMRL and Aurora's alignment posts mean you
  should be able to derive behaviour cloning's compounding error and explain how
  constrained RL or on-policy corrections fix it.
* **Model deployment and compilation** (quantization, operator coverage, per-hardware
  validation) is a first-class topic at Nuro because of licensing and at Aurora
  because of NVIDIA-based truck compute.
* **Structured safety argument**: if you have written a release gate or a regression
  contract for an OCR/detection product, frame it as claims and evidence; it is the
  language of Aurora's safety case.

## 6. Sources

**Zoox**

* Zoox journal, "Perception" — [zoox.com/journal/perception](https://zoox.com/journal/perception)
* Zoox journal, sensor staleness — [zoox.com/journal/sensor-staleness-zoox](https://zoox.com/journal/sensor-staleness-zoox)
* Zoox journal, public robotaxi service in Las Vegas (Sept 2025) — [zoox.com/journal/las-vegas](https://zoox.com/journal/las-vegas)
* Zoox journal, testing in Austin and Miami (2024) — [zoox.com/journal/austin-miami-2024/](https://zoox.com/journal/austin-miami-2024/); Seattle (2021) — [zoox.com/journal/seattle](https://zoox.com/journal/seattle)
* Zoox TeleGuidance and the Zoox safety report: cited by name (URLs not verified).

**Nuro**

* Nuro blog — [nuro.ai/blog](https://www.nuro.ai/blog)
* "CIMRL: Combining Imitation and Reinforcement Learning for Safe Autonomous Driving" — [blog post](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving), [arXiv:2406.08878](https://arxiv.org/abs/2406.08878)
* "Unified Perception Model" — [nuro.ai/blog/unified-perception-model](https://www.nuro.ai/blog/unified-perception-model)
* "FTL Model Compiler Framework" — [nuro.ai/blog/ftl-model-compiler-framework](https://www.nuro.ai/blog/ftl-model-compiler-framework)
* Nuro–Uber–Lucid robotaxi programme (2025): company announcements, cited by name.

**Aurora**

* Aurora — [aurora.tech](https://aurora.tech)
* "Aurora's Verifiable AI Approach to Self-Driving" (Chris Urmson, June 2024) — [blog.aurora.tech/engineering/aurora-verifiable-ai-approach-to-selfdriving](https://blog.aurora.tech/engineering/aurora-verifiable-ai-approach-to-selfdriving)
* "AI Alignment: Ensuring the Aurora Driver is Safe and Human-Like" (Drew Bagnell, 2024) — [blog.aurora.tech/engineering/the-future-of-ai-in-selfdriving](https://blog.aurora.tech/engineering/the-future-of-ai-in-selfdriving)
* "AI Transparency: The Why and How" (Drew Bagnell, 2024) — [blog.aurora.tech/engineering/the-why-and-how-of-transparency](https://blog.aurora.tech/engineering/the-why-and-how-of-transparency)
* "Aurora Unveils First-Ever Safety Case Framework" (2020) — [aurora.tech/blog/aurora-unveils-first-ever-safety-case-framework](https://aurora.tech/blog/aurora-unveils-first-ever-safety-case-framework); the framework — [safetycaseframework.aurora.tech](https://safetycaseframework.aurora.tech/)
* "FMCW Lidar: The Self-Driving Game-Changer" — [aurora.tech/newsroom/fmcw-lidar-the-self-driving-game-changer](https://aurora.tech/newsroom/fmcw-lidar-the-self-driving-game-changer); "FirstLight Lidar on a Chip" — [aurora.tech/newsroom/firstlight-lidar-on-a-chip](https://aurora.tech/newsroom/firstlight-lidar-on-a-chip)
* "The Road Never Sleeps: Aurora's Trucks Go Driverless Day and Night" (Aug 2025) — [aurora.tech/newsroom/…](https://aurora.tech/newsroom/the-road-never-sleeps-auroras-trucks-go-driverless-day-and-night); "Aurora Begins Commercial Driverless Trucking in Texas" (May 2025): newsroom, cited by name.
