# P3.4-16 is blocked: the rollout cannot yet drive a real hand

`QDGrasp-ContactRich-Tiny` was not generated. This records why, so the next
attempt starts from the diagnosis rather than repeating it.

## What was measured

Three probe iterations across all three release hands, against the
`generated_reachable` fixture -- the one geometry known to yield static
positives. **Zero dynamic positives in nine hand-iterations.** See
`probe-results.txt` for the counts and `probe_real_hand_rollout.py` for the
harness.

Each iteration fixed a real defect and exposed the next:

1. `grip` interpolated to the raw actuator `ctrlrange`, driving every joint to
   its stop simultaneously. That is a hand crushing itself, not a grasp, and it
   blew the safety budget on the first contact.
2. `grip` then interpolated the validated recipe's open and closed joint
   targets. Better, but LEAP reported a 6 km lift: wrist velocity was being
   added to actuators 0-2, which on a dexterous hand are finger joints, not a
   prismatic wrist.
3. The wrist moved to the welded mocap body and the hand was seeded at the
   recipe pregrasp pose. Still no positive, and LEAP still flies.

## Diagnosis

`run_static_seeded_rollout` reimplements a fraction of the `mocap-weld-v3`
protocol. The validated implementation is
`qdgrasp/dataset/pipeline/validators/mujoco_rollout.py::validate_grasp_rollout`,
which additionally carries weld settling, approach/squeeze/lift phasing at
pinned step counts, actuator gains read from the compiled MJCF, a contact noise
floor, and tracking-tolerance checks. Reproducing a fraction of a physics
protocol produces physics that looks plausible and is wrong.

## The fix, not attempted here

Phase 3.4 should **wrap** the validated rollout rather than reinvent it: feed it
primitive-derived joint targets, and layer the contact observer and safety
budget on top of its output. The parts that make Phase 3.4 different -- permitted
support and neighbour contact under a measured budget, and search over
trajectories -- sit above the protocol, not inside it.

That is a real refactor of `qdgrasp/dynamic/static_seeded.py` and a design
decision worth making deliberately. It was not made by continuing to patch the
simplified path, which is how the three iterations above were already spent.

## What this does not invalidate

The rollout works correctly on the micro scene it was built and tested against,
and every other Phase 3.4 module is unaffected: contracts, both backends, the
contact observer, primitives, the objective and reason ledger, CEM, refinement,
certification, storage and the ablation harness all pass on CPU. The gap is
specifically the control path from a primitive to a dexterous hand.

## SHA-256

- `probe-results.txt`: `7198ac67a8e4572fdfa1ea73424c78a785d6c7195fda03410d05f1572ec58721`
- `probe_real_hand_rollout.py`: `d32475c31fd485f94c9da10df8733a185e6cbefde9e0c66d185012532371a8ac`
