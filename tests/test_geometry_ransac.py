"""RANSAC: the iteration formula, and robust H / F / PnP fits under heavy contamination."""

import numpy as np

from mlbook.geometry import camera as C
from mlbook.geometry import ransac as RS


def _two_views(n: int = 80, seed: int = 0):
    """A rigid scene seen by two cameras with a wide baseline."""
    rng = np.random.default_rng(seed)
    K = C.intrinsics(800.0, 800.0, 640.0, 480.0)
    R1, t1 = C.look_at(np.array([0.0, -8.0, 6.0]), np.array([0.0, 0.0, 0.0]))
    R2, t2 = C.look_at(np.array([6.0, -6.0, 6.5]), np.array([0.0, 0.0, 0.0]))
    X = rng.uniform(-3, 3, size=(n, 3)) + np.array([0.0, 0.0, 0.5])  # (n, 3)
    p1, _ = C.project(X, K, R1, t1)  # (n, 2)
    p2, _ = C.project(X, K, R2, t2)  # (n, 2)
    return rng, K, (R1, t1), (R2, t2), X, p1, p2


def _contaminate(pts: np.ndarray, rng, frac: float = 0.3) -> np.ndarray:
    """Replace ``frac`` of the matches with uniformly random pixels (gross outliers)."""
    out = pts.copy()
    k = int(frac * len(pts))
    idx = rng.choice(len(pts), size=k, replace=False)
    out[idx] = rng.uniform(0, 960, size=(k, 2))
    return out, idx


def test_required_iterations_matches_the_closed_form():
    # k = log(1 - p) / log(1 - w^s); a homography at 50% inliers needs 72 draws
    assert RS.required_iterations(0.5, 4, 0.99) == 72
    # the eight-point algorithm pays for its larger minimal sample: 1177 draws
    assert RS.required_iterations(0.5, 8, 0.99) == 1177
    # clean data terminates immediately, and the count grows as inliers get rarer
    assert RS.required_iterations(1.0, 8) == 1
    assert RS.required_iterations(0.2, 4) > RS.required_iterations(0.6, 4)


def test_homography_dlt_is_exact_on_clean_planar_matches():
    rng = np.random.default_rng(1)
    H = np.array([[1.2, 0.1, 30.0], [-0.05, 0.9, -12.0], [1e-4, 2e-4, 1.0]])  # (3, 3)
    pts1 = rng.uniform(0, 800, size=(12, 2))  # (12, 2)
    pts2 = C.from_homogeneous(C.to_homogeneous(pts1) @ H.T)  # (12, 2)
    H_hat = RS.fit_homography_dlt(pts1, pts2)
    assert np.allclose(H_hat, H / H[2, 2], atol=1e-6)
    assert RS.transfer_error(H_hat, pts1, pts2).max() < 1e-6


def test_ransac_homography_survives_30_percent_outliers():
    rng = np.random.default_rng(2)
    H = np.array([[1.05, 0.08, 20.0], [-0.03, 0.97, 9.0], [5e-5, 1e-4, 1.0]])  # (3, 3)
    pts1 = rng.uniform(0, 900, size=(60, 2))  # (60, 2)
    pts2 = C.from_homogeneous(C.to_homogeneous(pts1) @ H.T) + rng.normal(0, 0.2, (60, 2))
    corrupted, bad = _contaminate(pts2, rng, frac=0.3)

    res = RS.ransac_homography(pts1, corrupted, threshold=3.0, rng=np.random.default_rng(3))

    assert not res.inliers[bad].any()          # no gross outlier voted for the model
    assert res.inliers.sum() >= 40             # nearly every true match was kept
    good = np.setdiff1d(np.arange(60), bad)
    assert RS.transfer_error(res.model, pts1[good], pts2[good]).max() < 3.0


def test_ransac_fundamental_recovers_the_epipolar_geometry():
    rng, K, _, _, _, p1, p2 = _two_views(n=100, seed=4)
    corrupted, bad = _contaminate(p2, rng, frac=0.25)

    res = RS.ransac_fundamental(p1, corrupted, threshold=1.0, max_iters=4000,
                                rng=np.random.default_rng(5))

    assert res.inliers.sum() >= 70
    assert res.inliers[bad].mean() < 0.05      # outliers essentially all rejected
    # the surviving matches satisfy x2^T F x1 = 0 to sub-pixel Sampson distance
    from mlbook.geometry.epipolar import sampson_distance
    assert sampson_distance(res.model, p1[res.inliers], corrupted[res.inliers]).max() < 1.0


def test_ransac_pnp_recovers_pose_with_mismatched_3d_points():
    rng, K, (R1, t1), _, X, p1, _ = _two_views(n=60, seed=6)
    corrupted, bad = _contaminate(p1, rng, frac=0.25)

    res = RS.ransac_pnp(X, corrupted, K, threshold=3.0, max_iters=3000,
                        rng=np.random.default_rng(7))
    R_hat, t_hat = res.model

    assert res.inliers.sum() >= 40 and res.inliers[bad].mean() < 0.05
    assert np.allclose(R_hat, R1, atol=1e-3)
    assert np.allclose(t_hat, t1, atol=1e-2)


def test_adaptive_stopping_cuts_iterations_on_clean_data():
    rng, K, _, _, _, p1, p2 = _two_views(n=60, seed=8)
    res = RS.ransac_fundamental(p1, p2, threshold=1.0, max_iters=2000,
                                rng=np.random.default_rng(9))
    # with ~100% inliers the first good sample collapses the budget to a handful of draws
    assert res.n_iters < 20
    assert res.inlier_ratio > 0.95
