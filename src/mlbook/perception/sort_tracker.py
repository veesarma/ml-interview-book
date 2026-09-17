"""SORT, DeepSORT's matching pieces, ByteTrack's two-stage association, and MOT metrics (NumPy).

SORT (Bewley et al., ICIP 2016): one Kalman filter per track with state
``[c_x, c_y, s, r, ċ_x, ċ_y, ṡ]`` (centre, area, aspect ratio and their rates; ``r`` has no
rate), predict every track one frame ahead, match predictions to detections with the
Hungarian algorithm on ``1 − IoU``, reject matches below ``iou_threshold``, spawn a track
per unmatched detection and kill tracks unseen for ``max_age`` frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlbook.perception.hungarian import hungarian
from mlbook.perception.kalman import KalmanFilter

# ---------------------------------------------------------------------------
# Box parameterisation and IoU
# ---------------------------------------------------------------------------


def bbox_to_z(box: np.ndarray) -> np.ndarray:
    """``(x1, y1, x2, y2)`` → measurement ``(c_x, c_y, s = w·h, r = w/h)``, shape (4,)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / max(h, 1e-9)])  # (4,)


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Inverse of ``bbox_to_z`` on the first four state entries: ``w = sqrt(s·r)``, ``h = s / w``."""
    w = np.sqrt(max(z[2] * z[3], 0.0))
    h = z[2] / max(w, 1e-9)
    return np.array([z[0] - w / 2.0, z[1] - h / 2.0, z[0] + w / 2.0, z[1] + h / 2.0])  # (4,)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) × (M, 4) → (N, M) IoU."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # (N, M, 2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2)
    wh = np.clip(rb - lt, 0.0, None)  # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]  # (N, M)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)  # (N, M)


# ---------------------------------------------------------------------------
# Association primitives (shared by SORT / DeepSORT / ByteTrack)
# ---------------------------------------------------------------------------


def associate(cost: np.ndarray, max_cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian on ``cost`` (N_tracks, N_dets), then drop pairs with ``cost > max_cost``.

    Returns ``matches (P, 2)`` as (track, det), ``unmatched_tracks (·,)``, ``unmatched_dets (·,)``.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=np.int64), np.arange(n), np.arange(m)
    rows, cols = hungarian(cost)
    keep = cost[rows, cols] <= max_cost  # (min(n,m),)
    matches = np.stack([rows[keep], cols[keep]], axis=1)  # (P, 2)
    unmatched_t = np.setdiff1d(np.arange(n), matches[:, 0])
    unmatched_d = np.setdiff1d(np.arange(m), matches[:, 1])
    return matches, unmatched_t, unmatched_d


def gated_appearance_cost(track_feats: np.ndarray, det_feats: np.ndarray, mahalanobis: np.ndarray,
                          gate: float = 9.4877) -> np.ndarray:
    """DeepSORT cost: cosine distance of appearance embeddings, gated by motion.

    Args:
        track_feats: (N, e) L2-normalised track embeddings (e.g. the mean of recent crops).
        det_feats: (M, e).  mahalanobis: (N, M) squared distances from each track's KF.
        gate: χ² 95 % quantile with 4 dof (the (c_x, c_y, s, r) measurement) = 9.4877.
    Returns:
        (N, M) with ``inf`` for gated-out pairs.
    """
    cos = 1.0 - track_feats @ det_feats.T  # (N, M) cosine distance
    return np.where(mahalanobis <= gate, cos, np.inf)  # (N, M)


def matching_cascade(cost: np.ndarray, time_since_update: np.ndarray, max_age: int, max_cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DeepSORT: match recently-seen tracks first so occluded tracks cannot steal detections.

    Iterates ``age = 0..max_age``; at each level solves the assignment for tracks with that
    ``time_since_update`` against the still-unmatched detections.
    """
    n, m = cost.shape
    matches: list[np.ndarray] = []
    free_dets = np.arange(m)
    for age in range(max_age + 1):
        tracks_at_age = np.flatnonzero(time_since_update == age)  # (·,)
        if tracks_at_age.size == 0 or free_dets.size == 0:
            continue
        sub = cost[np.ix_(tracks_at_age, free_dets)]  # (n_age, m_free)
        sub = np.where(np.isfinite(sub), sub, 1e6)  # Hungarian needs finite costs; inf pairs are then rejected
        mt, _, ud = associate(sub, max_cost)
        if mt.size:
            matches.append(np.stack([tracks_at_age[mt[:, 0]], free_dets[mt[:, 1]]], axis=1))
        free_dets = free_dets[ud]
    matched = np.concatenate(matches) if matches else np.zeros((0, 2), dtype=np.int64)  # (P, 2)
    unmatched_t = np.setdiff1d(np.arange(n), matched[:, 0])
    return matched, unmatched_t, free_dets


# ---------------------------------------------------------------------------
# SORT
# ---------------------------------------------------------------------------


def make_sort_kf(box: np.ndarray) -> KalmanFilter:
    """SORT's 7-state constant-velocity filter, with the original paper's noise choices."""
    F = np.eye(7)  # (7, 7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0  # c_x += ċ_x, c_y += ċ_y, s += ṡ
    H = np.zeros((4, 7))  # (4, 7)
    H[:4, :4] = np.eye(4)
    R = np.diag([1.0, 1.0, 10.0, 10.0])  # (4, 4) area / ratio are noisier measurements
    P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])  # (7, 7) unobserved velocities start very uncertain
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])  # (7, 7)
    x0 = np.zeros(7)  # (7,)
    x0[:4] = bbox_to_z(box)
    return KalmanFilter(F, H, Q, R, x0, P0)


@dataclass
class Track:
    kf: KalmanFilter
    track_id: int
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return z_to_bbox(self.kf.x[:4])  # (4,)


class SORT:
    """Frame-by-frame tracker.  ``update(dets (N, 5)) → (M, 5)`` rows ``x1 y1 x2 y2 id``."""

    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.tracks: list[Track] = []
        self.frame_count = 0
        self._next_id = 1

    def _predict_all(self) -> np.ndarray:
        """Advance every track one frame; returns predicted boxes (N_tracks, 4)."""
        preds = []
        for t in self.tracks:
            t.kf.predict()
            if t.kf.x[6] + t.kf.x[2] <= 0:  # area would go negative → freeze area rate
                t.kf.x[6] = 0.0
            t.age += 1
            t.time_since_update += 1
            preds.append(t.box)
        return np.stack(preds) if preds else np.zeros((0, 4))  # (N_tracks, 4)

    def _spawn(self, box: np.ndarray) -> None:
        self.tracks.append(Track(kf=make_sort_kf(box), track_id=self._next_id))
        self._next_id += 1

    def _apply_matches(self, matches: np.ndarray, dets: np.ndarray) -> None:
        for ti, di in matches:
            t = self.tracks[ti]
            t.kf.update(bbox_to_z(dets[di, :4]))
            t.hits += 1
            t.time_since_update = 0

    def _emit(self) -> np.ndarray:
        """Confirmed, currently-updated tracks; then prune dead ones."""
        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                out.append(np.concatenate([t.box, [t.track_id]]))  # (5,)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.stack(out) if out else np.zeros((0, 5))  # (M, 5)

    def update(self, dets: np.ndarray) -> np.ndarray:
        """One frame.  ``dets`` (N, 5) as x1 y1 x2 y2 score (N may be 0)."""
        self.frame_count += 1
        preds = self._predict_all()  # (N_tracks, 4)
        cost = 1.0 - iou_matrix(preds, dets[:, :4]) if len(dets) and len(preds) else np.zeros((len(preds), len(dets)))  # (N_tracks, N)
        matches, _, unmatched_d = associate(cost, 1.0 - self.iou_threshold)
        self._apply_matches(matches, dets)
        for di in unmatched_d:
            self._spawn(dets[di, :4])
        return self._emit()


class ByteTrack(SORT):
    """SORT with ByteTrack's second pass: leftover tracks try the *low-score* detections.

    Occluded objects keep producing low-confidence boxes; discarding them below a global
    threshold breaks tracks.  Pass 1: tracks ↔ high-score dets.  Pass 2: still-unmatched
    tracks ↔ low-score dets.  Only high-score leftovers may start a new track.
    """

    def __init__(self, high_thresh: float = 0.6, low_thresh: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.high_thresh, self.low_thresh = high_thresh, low_thresh

    def update(self, dets: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        preds = self._predict_all()  # (N_tracks, 4)
        high = dets[dets[:, 4] >= self.high_thresh] if len(dets) else np.zeros((0, 5))  # (N_h, 5)
        low = dets[(dets[:, 4] < self.high_thresh) & (dets[:, 4] >= self.low_thresh)] if len(dets) else np.zeros((0, 5))  # (N_l, 5)
        n_t = len(preds)
        cost_h = 1.0 - iou_matrix(preds, high[:, :4]) if n_t and len(high) else np.zeros((n_t, len(high)))  # (N_tracks, N_h)
        m1, ut1, ud_high = associate(cost_h, 1.0 - self.iou_threshold)
        self._apply_matches(m1, high)
        if ut1.size and len(low):
            cost_l = 1.0 - iou_matrix(preds[ut1], low[:, :4])  # (|ut1|, N_l)
            m2, _, _ = associate(cost_l, 1.0 - self.iou_threshold)
            if m2.size:
                self._apply_matches(np.stack([ut1[m2[:, 0]], m2[:, 1]], axis=1), low)
        for di in ud_high:
            self._spawn(high[di, :4])
        return self._emit()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mota(num_fn: int, num_fp: int, num_idsw: int, num_gt: int) -> float:
    """``MOTA = 1 − (FN + FP + IDSW) / GT``; can be negative; dominated by detection errors."""
    return 1.0 - (num_fn + num_fp + num_idsw) / max(num_gt, 1)


def idf1(idtp: int, idfp: int, idfn: int) -> float:
    """``IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)`` after a global identity assignment."""
    return 2.0 * idtp / max(2.0 * idtp + idfp + idfn, 1)


def hota_alpha(tp: int, fn: int, fp: int, tpa: np.ndarray, fna: np.ndarray, fpa: np.ndarray) -> float:
    """HOTA at one IoU threshold α: ``sqrt(DetA · AssA)``.

    ``DetA = TP/(TP+FN+FP)``; for each of the ``TP`` matched pairs ``c``,
    ``A(c) = TPA(c) / (TPA(c) + FNA(c) + FPA(c))`` counts how much of the two trajectories'
    lifetimes agree; ``AssA`` is the mean of ``A(c)``.  The final HOTA averages α over 0.05..0.95.
    """
    det_a = tp / max(tp + fn + fp, 1)
    ass = tpa / np.maximum(tpa + fna + fpa, 1)  # (TP,)
    ass_a = float(ass.mean()) if ass.size else 0.0
    return float(np.sqrt(det_a * ass_a))
