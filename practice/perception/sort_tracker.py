# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/sort_tracker.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k sort_tracker -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/sort_tracker --force

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

def bbox_to_z(box: np.ndarray) -> np.ndarray:
    """``(x1, y1, x2, y2)`` → measurement ``(c_x, c_y, s = w·h, r = w/h)``, shape (4,)."""
    raise NotImplementedError('TODO: implement bbox_to_z (see the reference in src/mlbook)')

def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Inverse of ``bbox_to_z`` on the first four state entries: ``w = sqrt(s·r)``, ``h = s / w``."""
    raise NotImplementedError('TODO: implement z_to_bbox (see the reference in src/mlbook)')

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) × (M, 4) → (N, M) IoU."""
    raise NotImplementedError('TODO: implement iou_matrix (see the reference in src/mlbook)')

def associate(cost: np.ndarray, max_cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian on ``cost`` (N_tracks, N_dets), then drop pairs with ``cost > max_cost``.

    Returns ``matches (P, 2)`` as (track, det), ``unmatched_tracks (·,)``, ``unmatched_dets (·,)``.
    """
    raise NotImplementedError('TODO: implement associate (see the reference in src/mlbook)')

def gated_appearance_cost(track_feats: np.ndarray, det_feats: np.ndarray, mahalanobis: np.ndarray, gate: float=9.4877) -> np.ndarray:
    """DeepSORT cost: cosine distance of appearance embeddings, gated by motion.

    Args:
        track_feats: (N, e) L2-normalised track embeddings (e.g. the mean of recent crops).
        det_feats: (M, e).  mahalanobis: (N, M) squared distances from each track's KF.
        gate: χ² 95 % quantile with 4 dof (the (c_x, c_y, s, r) measurement) = 9.4877.
    Returns:
        (N, M) with ``inf`` for gated-out pairs.
    """
    raise NotImplementedError('TODO: implement gated_appearance_cost (see the reference in src/mlbook)')

def matching_cascade(cost: np.ndarray, time_since_update: np.ndarray, max_age: int, max_cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DeepSORT: match recently-seen tracks first so occluded tracks cannot steal detections.

    Iterates ``age = 0..max_age``; at each level solves the assignment for tracks with that
    ``time_since_update`` against the still-unmatched detections.
    """
    raise NotImplementedError('TODO: implement matching_cascade (see the reference in src/mlbook)')

def make_sort_kf(box: np.ndarray) -> KalmanFilter:
    """SORT's 7-state constant-velocity filter, with the original paper's noise choices."""
    raise NotImplementedError('TODO: implement make_sort_kf (see the reference in src/mlbook)')

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
        raise NotImplementedError('TODO: implement box (see the reference in src/mlbook)')

class SORT:
    """Frame-by-frame tracker.  ``update(dets (N, 5)) → (M, 5)`` rows ``x1 y1 x2 y2 id``."""

    def __init__(self, max_age: int=1, min_hits: int=3, iou_threshold: float=0.3):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _predict_all(self) -> np.ndarray:
        """Advance every track one frame; returns predicted boxes (N_tracks, 4)."""
        raise NotImplementedError('TODO: implement _predict_all (see the reference in src/mlbook)')

    def _spawn(self, box: np.ndarray) -> None:
        raise NotImplementedError('TODO: implement _spawn (see the reference in src/mlbook)')

    def _apply_matches(self, matches: np.ndarray, dets: np.ndarray) -> None:
        raise NotImplementedError('TODO: implement _apply_matches (see the reference in src/mlbook)')

    def _emit(self) -> np.ndarray:
        """Confirmed, currently-updated tracks; then prune dead ones."""
        raise NotImplementedError('TODO: implement _emit (see the reference in src/mlbook)')

    def update(self, dets: np.ndarray) -> np.ndarray:
        """One frame.  ``dets`` (N, 5) as x1 y1 x2 y2 score (N may be 0)."""
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

class ByteTrack(SORT):
    """SORT with ByteTrack's second pass: leftover tracks try the *low-score* detections.

    Occluded objects keep producing low-confidence boxes; discarding them below a global
    threshold breaks tracks.  Pass 1: tracks ↔ high-score dets.  Pass 2: still-unmatched
    tracks ↔ low-score dets.  Only high-score leftovers may start a new track.
    """

    def __init__(self, high_thresh: float=0.6, low_thresh: float=0.1, **kwargs):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def update(self, dets: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

def mota(num_fn: int, num_fp: int, num_idsw: int, num_gt: int) -> float:
    """``MOTA = 1 − (FN + FP + IDSW) / GT``; can be negative; dominated by detection errors."""
    raise NotImplementedError('TODO: implement mota (see the reference in src/mlbook)')

def idf1(idtp: int, idfp: int, idfn: int) -> float:
    """``IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)`` after a global identity assignment."""
    raise NotImplementedError('TODO: implement idf1 (see the reference in src/mlbook)')

def hota_alpha(tp: int, fn: int, fp: int, tpa: np.ndarray, fna: np.ndarray, fpa: np.ndarray) -> float:
    """HOTA at one IoU threshold α: ``sqrt(DetA · AssA)``.

    ``DetA = TP/(TP+FN+FP)``; for each of the ``TP`` matched pairs ``c``,
    ``A(c) = TPA(c) / (TPA(c) + FNA(c) + FPA(c))`` counts how much of the two trajectories'
    lifetimes agree; ``AssA`` is the mean of ``A(c)``.  The final HOTA averages α over 0.05..0.95.
    """
    raise NotImplementedError('TODO: implement hota_alpha (see the reference in src/mlbook)')
