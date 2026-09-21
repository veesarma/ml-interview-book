# Part XXI: Zipline offboard perception

A targeted deep dive for one conversation: Zipline's Autonomy Droid Perception
engineer, offboard systems. It assumes you already build production vision systems
and that the part you are rusty on is metric 3-D geometry.

## What the role is asking for

The posting describes offboard (cloud) perception that supports onboard autonomy:
semantic segmentation of aerial imagery, feed-forward 4-D geometry, learned
preferences, 3-D and semantic priors shipped to the vehicle, classical geometry and
structure from motion alongside modern transformers, plus the training, data and
evaluation infrastructure behind all of it. Senior and staff both exist on the ladder,
and the staff bar is described in terms of owning a roadmap for one or more offboard
models and bringing up new efforts in stages.

Read that as a single sentence: *build the cloud-side system that decides what the
world looks like before a drone gets there, and make the decision auditable.*

Two consequences shape how you should answer everything.

**Offboard means a different cost curve.** No 10 ms latency budget, no embedded
power envelope. You can run structure from motion, an ensemble, a foundation model,
and a human in the loop. What you cannot do is ship a prior the vehicle will trust
and be wrong about it, because the vehicle has no way to check you.

**Priors are contracts.** A number the drone consumes (a ground height, a clear
radius, a deliverability score) is a promise with an error bar. Most of the staff-level
signal in this interview lives in how carefully you talk about that error bar.

## The shape of 45 minutes

Assume a working split of about 40% perception and ML system design, 25% geometry
fundamentals, 20% coding, 15% project deep dive. That is an inference from the job
description, and Zipline publishes no such split, so hold it loosely. A robotics
interviewer follows the thread you give them.

Which means you should give them a thread. Every answer in this part is written to
open a door you are ready to walk through.

## How to use this part in the time you have

=== "If you have two hours"

    1. [The cheatsheet](cheatsheet.md), once, out loud (15 min).
    2. [Two-view geometry](02-two-view-geometry.md) sections 1 and 2, deriving the
       epipolar constraint on paper (25 min).
    3. [SfM and bundle adjustment](03-sfm-and-bundle-adjustment.md) section 1, drawing
       the pipeline from memory until you can do it without looking (15 min).
    4. [The deliverability design](05-deliverability-system-design.md) end to end,
       speaking the answer aloud (30 min).
    5. [Coding drills](coding-drills.md): connected components and the landing-site
       question, typed from scratch (25 min).
    6. [The mock interview](mock-interview.md), read as a script (10 min).

=== "If you have a full day"

    Work chapters 1 to 6 in order, doing the "retype by hand" block in each one before
    reading the solution. Then the coding drills, then the mock, then
    [the story bank](story-bank.md) for your Google work. Budget the last hour for
    saying answers out loud instead of reading them.

=== "If you have fifteen minutes"

    [The cheatsheet](cheatsheet.md). Nothing else.

## The chapters

| Chapter | What it fixes |
|---|---|
| [1. Frames and camera geometry](01-frames-and-camera-geometry.md) | Homogeneous coordinates, SE(3), intrinsics and extrinsics, projection, distortion, and the frame conventions an aerial stack actually carries |
| [2. Two-view geometry](02-two-view-geometry.md) | Epipolar constraint, essential and fundamental matrices, RANSAC, triangulation, PnP, and when depth from triangulation is untrustworthy |
| [3. SfM and bundle adjustment](03-sfm-and-bundle-adjustment.md) | The full pipeline, the BA objective and its sparsity, gauge freedom, robust kernels, and the aerial-specific failure modes |
| [4. Learned 3-D and splats](04-learned-3d-and-splats.md) | Feed-forward pointmaps, what they buy over SfM and what they cost, Gaussian splatting, and why rendering quality is not geometric correctness |
| [5. Deliverability system design](05-deliverability-system-design.md) | The canonical question: is this backyard safe to deliver into? Geometry plus semantics plus a decision layer, staged as a roadmap |
| [6. Evaluation and scale](06-evaluation-and-scaling.md) | Why 97% accuracy and good mIoU can coexist with missing wires, safety-weighted slices, calibration, and going from 10<sup>4</sup> to 10<sup>7</sup> properties |
| [Coding drills](coding-drills.md) | IoU, NMS, connected components, landing sites, occupancy-grid search, bilinear sampling, with clarifying-question transcripts |
| [Mock interview](mock-interview.md) | A 45-minute transcript with drill-downs, plus a grading rubric |
| [Story bank](story-bank.md) | Translating Street View scale perception into aerial terms, with the follow-ups mapped |
| [Cheatsheet](cheatsheet.md) | One page, for the morning of |

## The code this part leans on

Everything derived here is implemented and tested in the repository, so you can run a
claim instead of trusting it.

| Module | What it holds |
|---|---|
| `src/mlbook/geometry/camera.py` | Intrinsics, extrinsics, projection, distortion |
| `src/mlbook/geometry/epipolar.py` | Eight-point algorithm, Sampson distance, epipoles |
| `src/mlbook/geometry/triangulation.py` | DLT triangulation, multi-view DLT, PnP |
| `src/mlbook/geometry/ransac.py` | Generic RANSAC, robust homography, fundamental matrix, PnP |
| `src/mlbook/geometry/bundle_adjustment.py` | LM bundle adjustment, Schur complement, Huber, ground control |
| `src/mlbook/geometry/sfm.py` | Essential decomposition, cheirality, incremental reconstruction |
| `src/mlbook/geometry/icp.py` | Umeyama, trimmed ICP, point-to-plane ICP |
| `src/mlbook/geometry/splatting.py` | Gaussian covariance parametrisation, EWA projection, compositing |
| `src/mlbook/perception/landing_zone.py` | Slope, relief, safety masks, connected components, exact EDT, ranked sites |
| `src/mlbook/perception/pointmap.py` | Feed-forward pointmaps, confidence loss, closed-form pose and focal |

Run them with `python -m pytest tests/test_geometry_ransac.py tests/test_geometry_bundle_adjustment.py tests/test_geometry_sfm.py tests/test_geometry_icp.py tests/test_geometry_splatting.py tests/test_perception_landing_zone.py tests/test_perception_pointmap.py -q`.

## Related reading in this book

* [Part IV, geometry](../part04-vision/06-geometry.md) and
  [3-D perception](../part04-vision/07-3d-perception.md) for the textbook treatment.
* [Part XI](../part11-perception-autonomy/index.md) for BEV, fusion, occupancy and tracking.
* [Part XVII, perception system design](../part17-ml-system-design/05-perception-system-av.md)
  for the generic AV framing this part specialises.
* [Waymo](../part18-company-deep-dives/waymo.md) and [Tesla](../part18-company-deep-dives/tesla.md)
  for how two other autonomy companies talk about evaluation and safety cases.
