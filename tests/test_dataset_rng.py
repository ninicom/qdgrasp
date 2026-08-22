from __future__ import annotations

import numpy as np

from qdgrasp.dataset.rng import (
    derive_seed,
    get_generator,
    sample_box_uniform,
    sample_quaternion_so3,
    sample_sphere_surface,
)


def test_derive_seed_deterministic_and_distinct() -> None:
    base = 42
    s1 = derive_seed(base, "train", "obj_0", 1)
    s2 = derive_seed(base, "train", "obj_0", 1)
    assert s1 == s2

    s3 = derive_seed(base, "train", "obj_0", 2)
    s4 = derive_seed(base, "val", "obj_0", 1)
    assert s1 != s3
    assert s1 != s4


def test_get_generator_reproducible_stream() -> None:
    rng1 = get_generator(12345, "split_a", 0)
    rng2 = get_generator(12345, "split_a", 0)

    v1 = rng1.uniform(0.0, 1.0, size=10)
    v2 = rng2.uniform(0.0, 1.0, size=10)
    np.testing.assert_array_equal(v1, v2)


def test_sample_quaternion_so3_unit_norm() -> None:
    rng = get_generator(999)
    q_single = sample_quaternion_so3(rng)
    assert q_single.shape == (4,)
    assert np.isclose(np.linalg.norm(q_single), 1.0, atol=1e-6)

    q_batch = sample_quaternion_so3(rng, size=50)
    assert q_batch.shape == (50, 4)
    norms = np.linalg.norm(q_batch, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_sample_sphere_surface_unit_norm() -> None:
    rng = get_generator(777)
    pts = sample_sphere_surface(rng, size=100)
    assert pts.shape == (100, 3)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_sample_box_uniform_bounds() -> None:
    rng = get_generator(888)
    bounds = [(-0.1, 0.1), (0.2, 0.5), (-1.0, -0.5)]
    pts = sample_box_uniform(rng, bounds=bounds, size=100)
    assert pts.shape == (100, 3)
    assert np.all(pts[:, 0] >= -0.1) and np.all(pts[:, 0] <= 0.1)
    assert np.all(pts[:, 1] >= 0.2) and np.all(pts[:, 1] <= 0.5)
    assert np.all(pts[:, 2] >= -1.0) and np.all(pts[:, 2] <= -0.5)
