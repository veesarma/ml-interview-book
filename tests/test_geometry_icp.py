"""Umeyama alignment and ICP: exactness with known correspondences, convergence without."""

import numpy as np

from mlbook.geometry import camera as C
from mlbook.geometry import icp as I


def _terrain(n: int = 250, seed: int = 0) -> np.ndarray:
    """A gently rolling ground patch: mostly planar with metre-scale relief."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-10, 10, size=(n, 2))                                   # (n, 2)
    z = 0.4 * np.sin(xy[:, 0] * 0.4) + 0.3 * np.cos(xy[:, 1] * 0.3)          # (n,)
    return np.column_stack([xy, z])                                          # (n, 3)


def _transform(seed: int = 1, angle: float = 0.15, scale: float = 1.0):
    R = C.rotation_from_axis_angle(np.array([0.2, -0.3, 1.0]), angle)        # (3, 3)
    t = np.array([1.5, -0.8, 0.6])                                           # (3,)
    return R, t, scale


def test_umeyama_is_exact_with_known_correspondences():
    src = _terrain(80, seed=2)
    R, t, _ = _transform()
    dst = I.apply_similarity(src, R, t)                                      # (80, 3)
    R_hat, t_hat, s_hat = I.kabsch_umeyama(src, dst)
    assert np.allclose(R_hat, R) and np.allclose(t_hat, t) and np.isclose(s_hat, 1.0)


def test_umeyama_recovers_scale_when_asked():
    src = _terrain(80, seed=3)
    R, t, _ = _transform()
    dst = I.apply_similarity(src, R, t, s=2.75)
    R_hat, t_hat, s_hat = I.kabsch_umeyama(src, dst, with_scale=True)
    assert np.isclose(s_hat, 2.75, atol=1e-9)
    assert np.allclose(I.apply_similarity(src, R_hat, t_hat, s_hat), dst, atol=1e-8)


def test_umeyama_never_returns_a_reflection():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(30, 3))
    dst = src * np.array([1.0, 1.0, -1.0])            # a mirror: no rotation can fit it
    R_hat, _, _ = I.kabsch_umeyama(src, dst)
    assert np.isclose(np.linalg.det(R_hat), 1.0)      # stayed in SO(3), so the fit is poor
    assert np.allclose(R_hat @ R_hat.T, np.eye(3), atol=1e-10)


def test_nearest_neighbours_and_normals():
    src = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    dst = np.array([[0.1, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    idx, dist = I.nearest_neighbours(src, dst)
    assert idx.tolist() == [0, 1]
    assert np.allclose(dist, [0.1, 1.0])

    rng = np.random.default_rng(5)
    plane = np.column_stack([rng.uniform(-5, 5, 60), rng.uniform(-5, 5, 60), np.zeros(60)])
    normals = I.estimate_normals(plane, k=8)          # (60, 3)
    assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1e-6)   # +/- z, sign unoriented


def test_icp_converges_from_a_coarse_initial_guess():
    src = _terrain(200, seed=6)
    R, t, _ = _transform(angle=0.12)
    dst = I.apply_similarity(src, R, t)
    rng = np.random.default_rng(7)
    dst = dst[rng.permutation(len(dst))]              # destroy the correspondence order

    res = I.icp(src, dst, max_iters=80)

    assert res.rmse < 1e-3
    assert np.allclose(res.R, R, atol=1e-3) and np.allclose(res.t, t, atol=1e-2)


def test_icp_with_scale_needs_a_coarse_initialisation():
    """Scale plus unknown correspondences is where ICP's non-convexity bites."""
    src = _terrain(150, seed=8)
    R, t, _ = _transform(angle=0.08)
    dst = I.apply_similarity(src, R, t, s=1.6)

    # from identity, nearest-neighbour assignment is wrong everywhere and ICP stalls
    naive = I.icp(src, dst, max_iters=100, with_scale=True)
    assert not np.isclose(naive.s, 1.6, rtol=0.1)

    # the standard fix: match centroids and the RMS radius before iterating
    s0 = float(np.sqrt((np.sum((dst - dst.mean(0)) ** 2, axis=1)).mean()
                       / (np.sum((src - src.mean(0)) ** 2, axis=1)).mean()))
    t0 = dst.mean(0) - s0 * src.mean(0)
    res = I.icp(src, dst, max_iters=100, with_scale=True, init=(np.eye(3), t0, s0))

    assert np.isclose(res.s, 1.6, rtol=2e-2)
    assert res.rmse < 5e-3


def test_trimmed_icp_ignores_points_outside_the_overlap():
    src = _terrain(160, seed=9)
    R, t, _ = _transform(angle=0.1)
    dst = I.apply_similarity(src[:120], R, t)         # target covers only 75% of the source
    rng = np.random.default_rng(10)
    dst = np.vstack([dst, rng.uniform(-40, 40, size=(20, 3))])   # plus junk far away

    full = I.icp(src, dst, max_iters=60, trim=1.0)
    trimmed = I.icp(src, dst, max_iters=60, trim=0.7)

    assert trimmed.rmse < full.rmse
    assert np.allclose(trimmed.R, R, atol=5e-3) and np.allclose(trimmed.t, t, atol=5e-2)


def test_point_to_plane_converges_fast_on_terrain():
    src = _terrain(140, seed=11)
    R, t, _ = _transform(angle=0.05)
    R_inv, t_inv = R.T, -R.T @ t
    dst = _terrain(140, seed=11)                      # same surface, different pose
    moved = I.apply_similarity(dst, R_inv, t_inv)     # source is the displaced copy

    res = I.icp_point_to_plane(moved, dst, max_iters=12)

    assert res.n_iters <= 12
    assert res.rmse < 0.05                            # sliding along the surface is free
