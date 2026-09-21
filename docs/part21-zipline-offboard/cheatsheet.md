# The morning-of cheatsheet

One page. Read it out loud once. Do not open anything else.

## Camera geometry

$$P_c = R P_w + t, \qquad C = -R^\top t, \qquad
T^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1\end{bmatrix}$$

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix},
\qquad u = f_x \frac{X_c}{Z_c} + c_x, \qquad v = f_y \frac{Y_c}{Z_c} + c_y$$

Order of operations: extrinsics, divide by $z$, **distortion**, then $K$.

$$\text{GSD} = \frac{h}{f_x}\ \text{m/px}. \qquad
\text{90 m, } f = 2400 \Rightarrow 3.75\ \text{cm/px}; \text{ a 1 m target is 27 px.}$$

Subscript discipline: $T_{w\leftarrow c} = T_{w\leftarrow b}\,T_{b\leftarrow c}$. Inner
subscripts cancel or the expression is wrong.

## Two-view geometry

$$x_2^\top E x_1 = 0,\quad E = [t]_\times R; \qquad
p_2^\top F p_1 = 0,\quad F = K_2^{-\top} E K_1^{-1}$$

* $E$: rank 2, singular values $(\sigma,\sigma,0)$, **5 dof**. Project onto that manifold after
  computing it from a noisy $F$.
* $F$: rank 2, $\det F = 0$, **7 dof**. Epipoles are the null vectors.
* Eight-point: $Af = 0$, smallest right singular vector, truncate to rank 2.
  **Hartley normalisation or it does not work.**
* Four poses from $E$; **cheirality** (positive depth in both cameras) picks one.
* Degeneracies: **pure rotation** ($t=0$, $E=0$) and **planar scene**. Fit $H$ too and compare.

$$\boxed{k = \frac{\log(1-p)}{\log(1-w^s)}}\qquad
w=0.5,\,p=0.99: \ s=4 \to 72,\ \ s=8 \to 1177,\ \ s=5 \to 146$$

$$\boxed{\sigma_Z \approx \frac{Z^2}{fB}\sigma_d}\qquad
90\ \text{m}, B=10\ \text{m}, f=2400, \sigma_d = 0.3\ \text{px} \Rightarrow \sigma_Z \approx 10\ \text{cm}$$

Depth error is **quadratic in range**, **linear in baseline**. Halve the baseline, double the
error.

Triangulation is unreliable from: short baseline, large depth, pose error, calibration error,
poor localisation, repeated texture, moving vegetation, water, occlusion.

## Bundle adjustment

$$\min_{\{R_i,t_i\},\{X_j\}} \sum_{(i,j)} \rho\big(\|\pi(K_i(R_iX_j + t_i)) - p_{ij}\|\big)$$

Update on the manifold: $R \leftarrow \exp([\delta\omega]_\times)R$, $t \leftarrow \exp([\delta\omega]_\times)t + \delta t$.

$$\frac{\partial Y}{\partial\delta\omega} = -[Y]_\times,\quad
\frac{\partial Y}{\partial\delta t} = I,\quad
\frac{\partial Y}{\partial X} = R,\quad
J_\pi = \begin{bmatrix}f_x & s\\0&f_y\end{bmatrix}\frac{1}{Z}\begin{bmatrix}1&0&-x\\0&1&-y\end{bmatrix}$$

Arrowhead Hessian. **Schur complement**:

$$(B - EC^{-1}E^\top)\delta_c = g_c - EC^{-1}g_p, \qquad \delta_p = C^{-1}(g_p - E^\top\delta_c)$$

500 cameras and 200k points: a $3000\times3000$ solve instead of $606{,}000$.

**Gauge freedom is 7 dof** (3 + 3 + 1 scale). Fix a camera, damp, or use ground control.

**Huber**: $w(e) = \min(1, \delta/e)$. At $e = 100$, $\delta = 2$: weight 0.02.

Scale comes from **GNSS baselines, altimeter, ground control points, known objects**. Never
from images.

## Learned 3-D

Pointmap: $(3, H, W)$ per view, both **in view 1's frame**. Depth is the $z$ channel; pose is
weighted Umeyama; focal has a closed form.

Loss: scale-normalise both sides, then $\sum (c_i e_i - \alpha\log c_i)$, optimum
$c^\star = \alpha/e$.

Splats: $(\mu, \Sigma, c, \alpha)$ with $\Sigma = RSS^\top R^\top$ (PSD by construction).
EWA: $\Sigma_{2D} = JW\Sigma W^\top J^\top + \epsilon I$.
Compositing: $C = \sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$, same as NeRF.

**Classical fails loudly** (a hole, which downstream treats as unknown).
**Learned fails quietly** (a confident wrong surface, which downstream accepts).
That asymmetry is the answer to most "which would you use" questions.

Never take metric geometry off a splat without independent validation. Rendering quality is
evidence about appearance.

## The deliverability design

1. **Start from what the vehicle consumes.** A prior, with an explicit unknown.
2. **Geometry** (surface, bare earth, slope, relief, height above ground, occupancy, confidence).
3. **Semantics** (surfaces and hazards, thin structures separately).
4. **Decision layer**, separate from both: a conjunction of hard constraints, then ranking.

$$\text{safe}(x) = [\text{slope}<\tau_s]\wedge[\text{clearance}>\tau_c]\wedge[P(\text{hazard})<\tau_o]\wedge[\text{corridor clear}]$$

* **Clearance is a radius.** Largest inscribed disk = max of the distance transform. Area is
  the wrong statistic (the L-shaped strip).
* **The corridor is vertical.** A tethered droid descends from altitude. Wires dominate.
* **Threshold $p + n\sigma$**, so uncertainty costs coverage instead of safety.
* **Deliver iff $P(\text{unsafe}) < C_{\text{decline}}/C_{\text{incident}}$.** Four orders of
  magnitude gives $10^{-4}$.
* **Three actions**: this site, an alternate site, decline and escalate.
* **Priors expire.** Validity horizon conditioned on what the site is made of.

Phases: rules and classical geometry, supervised semantics, learned multi-view geometry,
richer priors, continuous feedback. Each with a **promotion criterion set in advance**.

## Evaluation

A wire at 0.05% of pixels, missed entirely: **pixel accuracy 99.95%**, mIoU down 0.045,
**event recall 0**.

Hierarchy: mIoU, per-class IoU, rare-hazard recall, resolution-conditioned recall, slices,
end-to-end deliverability error, unsafe false-positive rate.

$$\boxed{p_{95\%} \approx 3/n}\qquad 10^{-4} \text{ needs } \approx 30{,}000 \text{ clean trials}$$

Split by **property**, never by tile. Calibrate, and check calibration **in the low-probability
regime**, per slice.

Offline win that does not transfer: distribution mismatch, wrong metric, leakage,
preprocessing difference, downstream trusting the new confidence differently, quiet-versus-loud
failure character.

## Scaling 10<sup>4</sup> to 10<sup>7</sup>

200 images per property: 2 billion images, ~8 PB raw.
Classical SfM at 30 core-min: **570 core-years**. Feed-forward at 0.5 GPU-s: **1400 GPU-hours**.

Changes: unit of work becomes a **tile** with a geospatial index; processing becomes
**incremental and change-driven**; the expensive path becomes **routed** with a random audit;
labelling becomes **sampling plus mining**; monitoring becomes **regional and change-based**;
and the **prior distribution system** is a real system with staged rollout and rollback.

## Complexity, for saying out loud

| Operation | Time | Note |
|---|---|---|
| Connected components (flood fill) | $O(HW)$ | $O(HW)$ worst-case space; iterate, do not recurse |
| Exact Euclidean distance transform | $O(HW)$ | Separable lower envelope, two passes |
| BFS on a grid | $O(HW)$ | Optimal only for uniform cost; mark on enqueue |
| Dijkstra | $O(HW\log HW)$ | Non-uniform cost; diagonals cost $\sqrt2$ |
| A* | same, fewer expansions | Needs an admissible heuristic; octile for 8-connectivity |
| NMS | $O(N^2)$, $O(N\log N)$ sort | Sort dominates when suppression is heavy |
| Bundle adjustment step | $O(C^3 + C^2P)$ | With Schur; $O((6C+3P)^3)$ without |
| Triangulation (DLT) | $O(V)$ per point | SVD of a $2V\times4$ matrix |

## The five answers, compressed

1. **20 images, metric 3-D.** Verify pairs by RANSAC on $F$ with a homography check; select
   pairs by triangulation angle rather than match count; initialise from INS; triangulate;
   bundle adjust with GNSS position priors and ground control for metric scale; hold out check
   points to report honest accuracy; densify to a surface model.
2. **Backyard safety.** Vehicle contract first, then geometry, semantics, decision layer as a
   logged conjunction, clearance as a radius, corridor not patch, upper confidence bound,
   staged roadmap.
3. **97% and misses wires.** Reject the metric, re-instrument at event level, diagnose
   resolution against imbalance against representation, then mitigate at the system level
   because detection recall alone will never be enough.
4. **Offline win, production loss.** Distribution, metric, leakage, integration, downstream
   calibration coupling, quiet-versus-loud failure character. Replay to isolate.
5. **10k to 10M.** The arithmetic first, then tiles, incremental, routed verification,
   sampling plus mining, regional monitoring, and the distribution of the priors.

## In the room

* Ask the **two or three questions that change the design**, then say how each one changed it.
* Say the **complexity before you type**.
* Write the **tests unprompted**.
* Name what is **loud** and what is **quiet** when you talk about failures.
* When you say "I would train X", say the **promotion criterion** in the next sentence.
* Four sentences, then **stop talking**.
