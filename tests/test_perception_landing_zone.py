"""Landing-zone analysis: derived geometry, connected components, the exact Euclidean
distance transform, clearance vs area, and the end-to-end site search over a synthetic yard."""

import numpy as np

from mlbook.perception import landing_zone as LZ


def _brute_force_edt(mask: np.ndarray) -> np.ndarray:
    """O(n^2) reference: distance from every True pixel to the nearest False pixel."""
    H, W = mask.shape
    bg = np.argwhere(~mask)                                   # (B, 2)
    out = np.zeros((H, W))
    if len(bg) == 0:
        return np.full((H, W), np.inf)
    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                out[y, x] = np.sqrt(((bg - np.array([y, x])) ** 2).sum(axis=1).min())
    return out


def _synthetic_yard(gsd: float = 0.10):
    """An 8 m x 8 m property: flat lawn, sloped driveway, a tree, a fence and a side path."""
    H = W = 80
    dsm = np.zeros((H, W))                                    # (80, 80) metres
    hazard = np.zeros((H, W))                                 # (80, 80) P(hazard)

    # driveway on the right third: a 20 degree ramp
    xs = np.arange(W)[None, :] * gsd                          # (1, 80) metres east
    ramp = np.tan(np.radians(20.0)) * (xs - 5.2)
    dsm[:, 52:] = np.repeat(ramp[:, 52:], H, axis=0)

    # a tree: 4 m tall canopy in the top-left corner
    yy, xx = np.mgrid[0:H, 0:W]
    tree = (yy - 12) ** 2 + (xx - 12) ** 2 < 9**2
    dsm[tree] = 4.0
    hazard[tree] = 0.95

    # a fence, then a narrow side path behind it: plenty of area, only 0.6 m wide
    dsm[72:74, :50] = 1.2
    dsm[74:, :50] = 0.02

    dtm = np.zeros((H, W))                                    # bare earth is flat here
    dtm[:, 52:] = dsm[:, 52:]                                 # the driveway *is* the ground
    return dsm, dtm, hazard


def test_slope_is_exact_on_a_synthetic_ramp():
    gsd = 0.25
    xs = np.arange(40)[None, :] * gsd                         # (1, 40)
    dsm = np.repeat(np.tan(np.radians(15.0)) * xs, 30, axis=0)  # (30, 40) 15 degree ramp
    slope = LZ.slope_degrees(dsm, gsd)                        # (30, 40)
    assert np.allclose(slope, 15.0, atol=1e-6)
    assert np.allclose(LZ.slope_degrees(np.zeros((10, 10)), gsd), 0.0)


def test_local_relief_finds_a_bump_a_slope_test_would_miss():
    dsm = np.zeros((20, 20))
    dsm[10, 10] = 0.4                                         # a single rock
    relief = LZ.local_relief(dsm, window=5)                   # (20, 20)
    assert np.isclose(relief[10, 10], 0.4)
    assert np.isclose(relief[8, 8], 0.4)                      # within the 5x5 window
    assert np.isclose(relief[0, 0], 0.0)                      # far away, flat


def test_height_above_ground_clips_reconstruction_noise():
    dsm = np.array([[1.0, 2.0], [0.5, 3.0]])
    dtm = np.array([[0.0, 0.0], [1.0, 0.0]])                  # DTM above DSM at (1, 0)
    hag = LZ.height_above_ground(dsm, dtm)
    assert np.allclose(hag, [[1.0, 2.0], [0.0, 3.0]])


def test_connected_components_respects_connectivity():
    mask = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],       # touches the block above only diagonally
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ], bool)
    labels4, n4 = LZ.connected_components(mask, connectivity=4)
    labels8, n8 = LZ.connected_components(mask, connectivity=8)

    assert n4 == 3                                            # the diagonal pixel is its own
    assert n8 == 1                                            # 8-connectivity chains them
    assert (labels4 > 0).sum() == mask.sum() == (labels8 > 0).sum()
    assert labels4[0, 0] == labels4[1, 1] and labels4[0, 0] != labels4[2, 2]


def test_connected_components_on_edge_cases():
    empty = np.zeros((4, 4), bool)
    labels, n = LZ.connected_components(empty)
    assert n == 0 and labels.max() == 0

    full = np.ones((4, 4), bool)
    labels, n = LZ.connected_components(full)
    assert n == 1 and (labels == 1).all()

    # a spiral is the worst case for the queue, and must still finish
    spiral = np.zeros((9, 9), bool)
    spiral[0, :] = spiral[:, -1] = spiral[-1, :] = True
    spiral[4, 1:6] = True
    spiral[1:5, 1] = True
    labels, n = LZ.connected_components(spiral)
    assert n == 1


def test_distance_transform_matches_brute_force():
    rng = np.random.default_rng(0)
    mask = rng.random((18, 22)) > 0.25                        # (18, 22) mostly True
    fast = LZ.distance_transform_edt(mask)
    slow = _brute_force_edt(mask)
    assert np.allclose(fast, slow, atol=1e-9)                 # exact, not a chamfer estimate

    # a 9x9 block of free space inside a border: the inradius is 5 pixels
    block = np.zeros((11, 11), bool)
    block[1:10, 1:10] = True
    assert np.isclose(LZ.distance_transform_edt(block).max(), 5.0)


def test_clearance_beats_area_on_an_l_shaped_region():
    """A long thin strip has plenty of area and nowhere to land."""
    strip = np.zeros((60, 60), bool)
    strip[5:55, 5:13] = True                                  # 50 x 8 px vertical arm
    strip[47:55, 5:55] = True                                 # 8 x 50 px horizontal arm
    square = np.zeros((60, 60), bool)
    square[20:40, 20:40] = True                               # 20 x 20 px

    assert strip.sum() > square.sum()                         # the strip has MORE area
    r_strip, _ = LZ.largest_inscribed_disk(strip)
    r_square, centre = LZ.largest_inscribed_disk(square)
    assert r_strip < r_square                                 # and LESS usable clearance
    assert np.isclose(r_square, 10.0)
    assert 29 <= centre[0] <= 30 and 29 <= centre[1] <= 30
    assert LZ.largest_inscribed_disk(np.zeros((5, 5), bool)) == (0.0, (0, 0))


def test_each_safety_constraint_rejects_on_its_own():
    cfg = LZ.SafetyConfig(gsd=0.1, hazard_dilation_m=0.0)
    ok = np.zeros((10, 10))
    safe, parts = LZ.safety_mask(ok, ok, ok, ok, cfg)
    assert safe.all()

    steep = np.full((10, 10), 30.0)
    assert not LZ.safety_mask(steep, ok, ok, ok, cfg)[0].any()
    rough = np.full((10, 10), 0.9)
    assert not LZ.safety_mask(ok, rough, ok, ok, cfg)[0].any()
    tall = np.full((10, 10), 2.0)
    assert not LZ.safety_mask(ok, ok, tall, ok, cfg)[0].any()
    hazardous = np.full((10, 10), 0.8)
    assert not LZ.safety_mask(ok, ok, ok, hazardous, cfg)[0].any()


def test_hazard_dilation_pushes_the_boundary_away_from_a_hazard():
    cfg = LZ.SafetyConfig(gsd=0.10, hazard_dilation_m=0.50, max_slope_deg=90.0)
    flat = np.zeros((40, 40))
    hazard = np.zeros((40, 40))
    hazard[20, 20] = 1.0                                      # one hazardous pixel

    safe, _ = LZ.safety_mask(flat, flat, flat, hazard, cfg)

    assert not safe[20, 20]
    assert not safe[20, 23]                                   # 3 px = 0.30 m, inside the buffer
    assert safe[20, 26]                                       # 6 px = 0.60 m, outside it


def test_find_landing_sites_picks_the_lawn_and_rejects_the_rest():
    dsm, dtm, hazard = _synthetic_yard()
    cfg = LZ.SafetyConfig(gsd=0.10, min_clearance_m=0.75, max_slope_deg=10.0,
                          max_relief_m=0.25, hazard_dilation_m=0.5)

    sites, safe, parts = LZ.find_landing_sites(dsm, dtm, hazard, cfg, min_area_m2=1.0)

    assert len(sites) >= 1
    best = sites[0]
    assert best.clearance_m >= cfg.min_clearance_m
    assert best.mean_slope_deg < cfg.max_slope_deg
    # the chosen touchdown point must be clear of the tree and off the driveway
    row, col = best.centre_px
    assert hazard[row, col] < 0.05
    assert col < 52
    assert np.hypot(row - 12, col - 12) > 9
    # the steep driveway and the tree canopy were both excluded from the mask
    assert not safe[:, 60:].any()
    assert not safe[12, 12]
    assert not parts["slope"][:, 60:].any()


def test_narrow_path_has_area_but_never_becomes_a_site():
    dsm, dtm, hazard = _synthetic_yard()
    cfg = LZ.SafetyConfig(gsd=0.10, min_clearance_m=1.5, hazard_dilation_m=0.0)
    sites, safe, _ = LZ.find_landing_sites(dsm, dtm, hazard, cfg, min_area_m2=0.5)

    # the 0.8 m strip along the bottom is flat and passes every pixel-wise test
    assert safe[76, 10]
    # but no site is centred there, because no 1.5 m disk fits
    assert all(s.centre_px[0] < 74 for s in sites)


def test_site_score_prefers_more_clearance_and_flatter_ground():
    cfg = LZ.SafetyConfig()
    base = dict(label=1, centre_px=(0, 0), centre_m=(0.0, 0.0), area_m2=9.0,
                mean_slope_deg=2.0, max_relief_m=0.05, mean_hazard_prob=0.0)
    roomy = LZ.site_score(LZ.LandingSite(clearance_m=3.0, **base), cfg)
    tight = LZ.site_score(LZ.LandingSite(clearance_m=1.6, **base), cfg)
    steep = LZ.site_score(LZ.LandingSite(clearance_m=3.0, **{**base, "mean_slope_deg": 9.0}), cfg)

    assert roomy > tight and roomy > steep
    assert 0.0 <= tight <= 1.0 and 0.0 <= roomy <= 1.0


def test_risk_adjusted_mask_spends_uncertainty_on_coverage_not_safety():
    prob = np.array([[0.02, 0.02, 0.20]])
    std = np.array([[0.005, 0.10, 0.005]])
    keep = LZ.risk_adjusted_mask(prob, std, budget=0.05, n_sigma=2.0)
    # confident-and-clean passes; equally clean but uncertain does not
    assert keep.tolist() == [[True, False, False]]
    assert LZ.risk_adjusted_mask(prob, np.zeros_like(std), 0.05).tolist() == [[True, True, False]]
