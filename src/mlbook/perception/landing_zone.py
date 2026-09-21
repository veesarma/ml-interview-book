"""From an aerial surface model to ranked delivery sites (NumPy).

This is the offboard question in its most concrete form: given what a survey flight and a
segmentation model produced over one property, where (if anywhere) can a drone put a
package down, and how sure are we?

    DSM + DTM           per-pixel geometry (surface height, ground height)
    semantic hazard     per-pixel P(hazard): wires, trees, water, vehicles, people
            |
            v
    derived geometry    slope, local relief, height above ground
            |
            v
    safety mask         a conjunction of hard constraints, each independently checkable
            |
            v
    connected components + distance transform
            |
            v
    ranked landing sites with a clearance radius and a calibrated score

Two design commitments are worth defending out loud:

* **The segmentation network does not answer "can I deliver".**  It answers "what is this
  pixel".  Deliverability is a separate decision layer over geometry, semantics and
  history, which keeps the safety constraints auditable and lets you change a threshold
  without retraining anything.
* **Clearance is a radius, not an area.**  A 20 m^2 L-shaped strip between a fence and a
  shed has plenty of area and nowhere to put a 1.5 m disk.  The largest inscribed circle,
  which the distance transform gives exactly, is the quantity that matters.

Conventions: rasters are ``(H, W)`` with north up, ``gsd`` is ground sample distance in
metres per pixel, and heights are metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ derived geometry


def slope_degrees(dsm: np.ndarray, gsd: float) -> np.ndarray:
    """Terrain slope from a surface model, in degrees. (H, W) -> (H, W).

    Central differences give the gradient in metres per metre; the slope is the angle of
    the steepest descent direction, ``atan(sqrt(dz/dx^2 + dz/dy^2))``.  Edges fall back to
    one-sided differences, which is what ``np.gradient`` already does.
    """
    dz_dy, dz_dx = np.gradient(dsm, gsd)                 # (H, W) each, metres per metre
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))  # (H, W)


def _sliding_extrema(a: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Separable sliding-window min and max with edge padding. (H, W) -> two (H, W).

    Running the 1-D window down the rows and then across the columns costs ``O(H W k)``
    instead of the ``O(H W k^2)`` of a 2-D window, because max is associative and
    separable over a rectangle.
    """
    r = window // 2
    out_max, out_min = a, a
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        pm = np.pad(out_max, pad, mode="edge")
        pn = np.pad(out_min, pad, mode="edge")
        wins_max = np.lib.stride_tricks.sliding_window_view(pm, window, axis=axis)
        wins_min = np.lib.stride_tricks.sliding_window_view(pn, window, axis=axis)
        out_max = wins_max.max(axis=-1)                  # (H, W)
        out_min = wins_min.min(axis=-1)                  # (H, W)
    return out_min, out_max


def local_relief(dsm: np.ndarray, window: int = 5) -> np.ndarray:
    """Peak-to-peak height inside a square window: the roughness a landing gear feels.

    Slope alone passes a field of small rocks or a stepped patio; relief catches them.
    The two constraints are complementary and both belong in the mask.
    """
    lo, hi = _sliding_extrema(dsm, window)               # (H, W), (H, W)
    return hi - lo                                       # (H, W) metres


def height_above_ground(dsm: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    """Canopy / object height: surface model minus bare-earth model. (H, W) -> (H, W).

    Negative values mean the DTM sits above the DSM, which is a reconstruction error
    rather than a hole in the ground, so they are clipped at zero.
    """
    return np.maximum(dsm - dtm, 0.0)                    # (H, W)


# ---------------------------------------------------------------------- safety mask


@dataclass
class SafetyConfig:
    """Hard constraints for a delivery site.  Every field is a number a reviewer can argue with."""

    gsd: float = 0.10                  # metres per pixel
    max_slope_deg: float = 10.0        # steeper than this and the package rolls
    max_relief_m: float = 0.25         # peak-to-peak inside the relief window
    relief_window: int = 5             # pixels
    max_height_above_ground_m: float = 0.30   # anything taller is an obstacle
    max_hazard_prob: float = 0.05      # per-pixel semantic hazard budget
    min_clearance_m: float = 1.50      # radius of the disk that must fit
    hazard_dilation_m: float = 0.50    # keep this far away from any hazard pixel


def safety_mask(
    slope: np.ndarray, relief: np.ndarray, hag: np.ndarray,
    hazard_prob: np.ndarray, cfg: SafetyConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Conjunction of the hard constraints, plus the per-constraint masks for debugging.

    Returning the individual masks is not decoration.  When the system rejects a yard the
    first question is always *which* constraint rejected it, and a single boolean cannot
    answer that.

    Returns:
        safe (H, W) bool, and a dict of the four component masks.
    """
    parts = {
        "slope": slope <= cfg.max_slope_deg,                         # (H, W) bool
        "relief": relief <= cfg.max_relief_m,                        # (H, W)
        "height": hag <= cfg.max_height_above_ground_m,              # (H, W)
        "semantic": hazard_prob <= cfg.max_hazard_prob,              # (H, W)
    }
    safe = parts["slope"] & parts["relief"] & parts["height"] & parts["semantic"]
    if cfg.hazard_dilation_m > 0:
        # push the boundary back from every hazard: a buffer around wires and trunks
        margin_px = cfg.hazard_dilation_m / cfg.gsd
        dist_to_hazard = distance_transform_edt(parts["semantic"])   # (H, W) pixels
        safe = safe & (dist_to_hazard >= margin_px)
    return safe, parts


# ------------------------------------------------------- connected components + EDT


def connected_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Label connected regions of ``True`` with an iterative flood fill.

    Breadth-first from every unvisited foreground pixel.  Each pixel enters the queue once,
    so the cost is ``O(H W)`` time and ``O(H W)`` worst-case auxiliary space (a single row
    of a spiral fills the queue).  The loop is iterative on purpose: recursion blows the
    Python stack on any real raster.

    Args:
        mask: (H, W) bool.  connectivity: 4 or 8.
    Returns:
        labels (H, W) int32 with 0 as background and 1..n as regions, and n.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)                  # (H, W) 0 = background
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
                y, x = queue.pop()                        # LIFO: a stack, so this is DFS
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = n
                        queue.append((ny, nx))
    return labels, n


def _edt_1d(f: np.ndarray) -> np.ndarray:
    """Felzenszwalb-Huttenlocher lower envelope: ``D(x) = min_y (x - y)^2 + f(y)``. (n,) -> (n,).

    The parabolas ``(x - y)^2 + f(y)`` have a lower envelope with at most ``n`` pieces, and
    one left-to-right sweep maintaining the current envelope finds them all in ``O(n)``.
    """
    n = len(f)
    v = np.zeros(n, np.int64)                            # (n,) parabola indices in the envelope
    z = np.empty(n + 1)                                  # (n+1,) envelope breakpoints
    z[0], z[1], k = -np.inf, np.inf, 0
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:                                 # this parabola hides the last one
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
    """Exact Euclidean distance, in pixels, from each ``True`` pixel to the nearest ``False``.

    Two separable ``O(H W)`` passes of the 1-D lower-envelope transform, columns then rows.
    Exact, unlike the 3-4 chamfer approximation, which matters when the number feeds a
    clearance constraint measured in metres.

    Returns:
        (H, W) float distances; ``0`` on background pixels.
    """
    INF = 1e12
    f = np.where(mask, INF, 0.0)                         # (H, W) seed: 0 at background
    d = np.apply_along_axis(_edt_1d, 0, f)               # (H, W) squared distance down columns
    d = np.apply_along_axis(_edt_1d, 1, d)               # (H, W) then across rows
    return np.sqrt(d)                                    # (H, W) pixels


# ------------------------------------------------------------------- region analysis


@dataclass
class LandingSite:
    """One candidate site, with everything a reviewer needs to accept or reject it."""

    label: int
    centre_px: tuple[int, int]        # (row, col) of the best touchdown point
    centre_m: tuple[float, float]     # (north, east) offset from the raster origin
    clearance_m: float                # radius of the largest disk that fits, in metres
    area_m2: float
    mean_slope_deg: float
    max_relief_m: float
    mean_hazard_prob: float
    score: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # (y0, x0, y1, x1), half-open


def region_properties(labels: np.ndarray, n: int, gsd: float) -> list[dict]:
    """Area, bounding box and centroid for every label. ``O(H W + n)`` with one pass.

    ``np.bincount`` over the flattened labels does all three at once, which beats looping
    over ``n`` masks when the raster has hundreds of components.
    """
    flat = labels.ravel()                                # (H*W,)
    ys, xs = np.divmod(np.arange(labels.size), labels.shape[1])
    counts = np.bincount(flat, minlength=n + 1)[1:]      # (n,) pixels per label
    sum_y = np.bincount(flat, weights=ys, minlength=n + 1)[1:]   # (n,)
    sum_x = np.bincount(flat, weights=xs, minlength=n + 1)[1:]   # (n,)
    out = []
    for i in range(n):
        m = labels == i + 1                              # (H, W) bool
        yy, xx = np.nonzero(m)
        out.append({
            "label": i + 1,
            "area_px": int(counts[i]),
            "area_m2": float(counts[i] * gsd * gsd),
            "centroid_px": (sum_y[i] / counts[i], sum_x[i] / counts[i]),
            "bbox": (int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1),
        })
    return out


def largest_inscribed_disk(mask: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Radius (pixels) and centre of the largest disk fitting inside ``mask``.

    The distance transform *is* the answer: its maximum is the inradius and its argmax is
    the deepest interior point, the Chebyshev centre of the region.

    Returns:
        (radius_px, (row, col)).  Radius 0 for an empty mask.
    """
    if not mask.any():
        return 0.0, (0, 0)
    dist = distance_transform_edt(mask)                  # (H, W)
    flat = int(np.argmax(dist))
    row, col = np.unravel_index(flat, mask.shape)
    return float(dist[row, col]), (int(row), int(col))


def find_landing_sites(
    dsm: np.ndarray, dtm: np.ndarray, hazard_prob: np.ndarray, cfg: SafetyConfig,
    min_area_m2: float = 1.0,
) -> tuple[list[LandingSite], np.ndarray, dict[str, np.ndarray]]:
    """The whole offboard pass: rasters in, ranked sites out.

    Args:
        dsm: (H, W) surface height in metres.  dtm: (H, W) bare-earth height.
        hazard_prob: (H, W) per-pixel P(hazard) from the semantic model.
        min_area_m2: drop components smaller than this before the expensive per-region work.

    Returns:
        sites sorted best first, the safety mask (H, W), and the per-constraint masks.
    """
    slope = slope_degrees(dsm, cfg.gsd)                              # (H, W)
    relief = local_relief(dsm, cfg.relief_window)                    # (H, W)
    hag = height_above_ground(dsm, dtm)                              # (H, W)
    safe, parts = safety_mask(slope, relief, hag, hazard_prob, cfg)  # (H, W), dict
    labels, n = connected_components(safe, connectivity=4)

    sites: list[LandingSite] = []
    for prop in region_properties(labels, n, cfg.gsd):
        if prop["area_m2"] < min_area_m2:
            continue
        region = labels == prop["label"]                             # (H, W) bool
        radius_px, centre = largest_inscribed_disk(region)
        clearance_m = radius_px * cfg.gsd
        if clearance_m < cfg.min_clearance_m:
            continue                                                 # no disk of the required size
        sites.append(LandingSite(
            label=prop["label"],
            centre_px=centre,
            centre_m=(centre[0] * cfg.gsd, centre[1] * cfg.gsd),
            clearance_m=clearance_m,
            area_m2=prop["area_m2"],
            mean_slope_deg=float(slope[region].mean()),
            max_relief_m=float(relief[region].max()),
            mean_hazard_prob=float(hazard_prob[region].mean()),
            bbox=prop["bbox"],
        ))
    for s in sites:
        s.score = site_score(s, cfg)
    sites.sort(key=lambda s: -s.score)
    return sites, safe, parts


def site_score(site: LandingSite, cfg: SafetyConfig) -> float:
    """A transparent ranking score in [0, 1] over sites that already passed every constraint.

    Deliberately a weighted product of normalised margins rather than a learned scalar:
    ranking happens *after* the hard constraints, so the score only has to order feasible
    options, and a product means one near-zero margin cannot be averaged away by the others.
    A learned model replaces this once you have outcome labels from real deliveries.
    """
    clearance = min(site.clearance_m / (2.0 * cfg.min_clearance_m), 1.0)
    flatness = max(0.0, 1.0 - site.mean_slope_deg / cfg.max_slope_deg)
    smoothness = max(0.0, 1.0 - site.max_relief_m / cfg.max_relief_m)
    semantic = max(0.0, 1.0 - site.mean_hazard_prob / cfg.max_hazard_prob)
    return float(clearance * (0.25 + 0.75 * flatness) * (0.25 + 0.75 * smoothness)
                 * (0.25 + 0.75 * semantic))


def risk_adjusted_mask(
    hazard_prob: np.ndarray, hazard_std: np.ndarray, budget: float, n_sigma: float = 2.0
) -> np.ndarray:
    """Threshold the *upper confidence bound* of hazard probability, not the mean.

    A model that says "5% hazard, plus or minus 20%" and a model that says "5% hazard, plus
    or minus 1%" should not produce the same decision.  Using ``p + n_sigma * sigma``
    makes an uncertain pixel fail the constraint, so uncertainty costs coverage rather than
    safety, which is the direction a delivery system wants to be wrong in.

    Args:
        hazard_prob: (H, W) predictive mean.  hazard_std: (H, W) predictive std,
            for example from an ensemble or MC dropout.
    Returns:
        (H, W) bool, True where the pixel is acceptable even under the pessimistic estimate.
    """
    return (hazard_prob + n_sigma * hazard_std) <= budget            # (H, W) bool
