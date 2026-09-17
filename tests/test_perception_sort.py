import numpy as np

from mlbook.perception import sort_tracker as st


def test_bbox_z_round_trip_and_iou():
    box = np.array([10.0, 20.0, 50.0, 100.0])
    assert np.allclose(st.z_to_bbox(st.bbox_to_z(box)), box)
    iou = st.iou_matrix(box[None], np.array([[10.0, 20.0, 50.0, 100.0], [0.0, 0.0, 5.0, 5.0]]))
    assert np.allclose(iou, [[1.0, 0.0]])


def test_associate_rejects_high_cost_pairs():
    cost = np.array([[0.1, 0.9], [0.95, 0.2], [0.5, 0.5]])
    m, ut, ud = st.associate(cost, max_cost=0.3)
    assert m.tolist() == [[0, 0], [1, 1]] and ut.tolist() == [2] and ud.size == 0


def test_gated_appearance_cost_and_cascade():
    tf = np.eye(3)[:2]  # two tracks with orthogonal appearance
    df = np.eye(3)[[1, 0]]  # detections in swapped order
    maha = np.array([[1.0, 1.0], [1.0, 50.0]])  # track 1 ↔ det 1 gated out by motion
    cost = st.gated_appearance_cost(tf, df, maha)
    assert cost[0, 1] == 0.0 and np.isinf(cost[1, 1]) and cost[1, 0] == 0.0
    # cascade: the fresh track (age 0) gets first pick even if an older track would score better
    c = np.array([[0.2, 0.9], [0.1, 0.9]])
    m, ut, ud = st.matching_cascade(c, time_since_update=np.array([1, 0]), max_age=2, max_cost=0.5)
    assert m.tolist() == [[1, 0]] and ut.tolist() == [0] and ud.tolist() == [1]


def _moving_boxes(n_frames, n_obj, rng):
    frames = []
    starts = rng.uniform(0, 300, size=(n_obj, 2))
    vel = rng.uniform(-3, 3, size=(n_obj, 2))
    for t in range(n_frames):
        xy = starts + t * vel
        frames.append(np.concatenate([xy, xy + 40.0, np.full((n_obj, 1), 0.9)], axis=1))  # (n_obj, 5)
    return frames


def test_sort_keeps_identities_on_smooth_motion():
    rng = np.random.default_rng(0)
    tracker = st.SORT(max_age=2, min_hits=2, iou_threshold=0.3)
    frames = _moving_boxes(20, 4, rng)
    ids_per_obj = [set() for _ in range(4)]
    for dets in frames:
        out = tracker.update(dets)
        for row in out:
            j = int(np.argmin(np.abs(dets[:, 0] - row[0]) + np.abs(dets[:, 1] - row[1])))
            ids_per_obj[j].add(int(row[4]))
    assert all(len(s) == 1 for s in ids_per_obj)  # no ID switches
    assert len({next(iter(s)) for s in ids_per_obj}) == 4  # four distinct ids


def test_sort_birth_and_death():
    tracker = st.SORT(max_age=1, min_hits=1)
    box = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]])
    assert tracker.update(box).shape == (1, 5)
    assert tracker.update(np.zeros((0, 5))).shape == (0, 5)  # missed once: coasting, not emitted
    assert len(tracker.tracks) == 1
    tracker.update(np.zeros((0, 5)))  # missed twice > max_age → deleted
    assert len(tracker.tracks) == 0


def test_sort_survives_a_missed_frame_without_new_id():
    tracker = st.SORT(max_age=2, min_hits=1)
    frames = _moving_boxes(8, 1, np.random.default_rng(1))
    ids = []
    for t, dets in enumerate(frames):
        out = tracker.update(np.zeros((0, 5)) if t == 3 else dets)
        ids += [int(r[4]) for r in out]
    assert set(ids) == {1}


def test_bytetrack_recovers_track_from_low_score_detection():
    frames = _moving_boxes(10, 1, np.random.default_rng(2))
    for t in (4, 5):
        frames[t][:, 4] = 0.3  # occluded: score drops below the high threshold
    sort_ids, byte_ids = [], []
    sort = st.SORT(max_age=1, min_hits=1)
    byte = st.ByteTrack(high_thresh=0.6, low_thresh=0.1, max_age=1, min_hits=1)
    for dets in frames:
        high_only = dets[dets[:, 4] >= 0.6]
        sort_ids += [int(r[4]) for r in sort.update(high_only)]
        byte_ids += [int(r[4]) for r in byte.update(dets)]
    assert set(byte_ids) == {1} and len(byte_ids) == 10  # ByteTrack never loses the object
    assert len(set(sort_ids)) == 2  # thresholded SORT dropped the track and re-spawned it


def test_mot_metrics():
    assert np.isclose(st.mota(num_fn=10, num_fp=5, num_idsw=1, num_gt=100), 0.84)
    assert np.isclose(st.idf1(idtp=80, idfp=20, idfn=20), 0.8)
    h = st.hota_alpha(tp=10, fn=0, fp=0, tpa=np.full(10, 5), fna=np.zeros(10), fpa=np.full(10, 5))
    assert np.isclose(h, np.sqrt(1.0 * 0.5))
