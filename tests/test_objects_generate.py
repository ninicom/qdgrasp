from __future__ import annotations

from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.generate import (
    generate_box,
    generate_capsule,
    generate_compound_convex,
    generate_cylinder,
    generate_sphere,
    generate_superquadric,
)
from qdgrasp.objects.manifest import export_mesh_deterministic_obj, sha256_bytes


def test_primitive_generators_bit_exact_reproducibility() -> None:
    for gen_func in (generate_box, generate_sphere, generate_cylinder, generate_capsule):
        rng1 = get_generator(42, "obj_test")
        m1, g1, p1, mass1, in1 = gen_func(rng1)

        rng2 = get_generator(42, "obj_test")
        m2, g2, p2, mass2, in2 = gen_func(rng2)

        obj1 = export_mesh_deterministic_obj(m1)
        obj2 = export_mesh_deterministic_obj(m2)

        assert sha256_bytes(obj1) == sha256_bytes(obj2)
        assert mass1 == mass2
        assert in1 == in2
        assert p1 == p2
        assert len(g1) == len(g2)
        assert mass1 > 0.0
        assert all(i > 0 for i in in1)


def test_superquadric_generator_reproducibility() -> None:
    rng1 = get_generator(101, "sq_test")
    m1, _g1, _p1, mass1, in1 = generate_superquadric(rng1)

    rng2 = get_generator(101, "sq_test")
    m2, _g2, _p2, mass2, in2 = generate_superquadric(rng2)

    obj1 = export_mesh_deterministic_obj(m1)
    obj2 = export_mesh_deterministic_obj(m2)

    assert sha256_bytes(obj1) == sha256_bytes(obj2)
    assert mass1 == mass2
    assert in1 == in2
    assert len(m1.vertices) > 0
    assert len(m1.faces) > 0


def test_compound_convex_generators_reproducibility() -> None:
    for family in ("t_shape", "l_shape", "dumbbell"):
        rng1 = get_generator(202, family)
        m1, g1, _p1, mass1, in1 = generate_compound_convex(rng1, shape_family=family)

        rng2 = get_generator(202, family)
        m2, _g2, _p2, _mass2, _in2 = generate_compound_convex(rng2, shape_family=family)

        obj1 = export_mesh_deterministic_obj(m1)
        obj2 = export_mesh_deterministic_obj(m2)

        assert sha256_bytes(obj1) == sha256_bytes(obj2)
        assert len(g1) >= 2  # Compound objects have multiple convex sub-geoms
        assert mass1 > 0.0
        assert all(i > 0 for i in in1)
