# P3.4-04 backend compatibility spike

The MJX / MuJoCo-Warp decision needs a concrete list of what the release models
actually use. `phase3_4_backend_spike.py` builds the real rollout model for each
hand and enumerates it, so the decision is not made on a general impression that
"MJX supports most of MuJoCo".

Produced on CPU with MuJoCo 3.12.0. No GPU required.

## Per-hand model features

| hand | actuators | tendons | equality | mocap | integrator | per-contact force readable |
| --- | --- | --- | --- | --- | --- | --- |
| `leap_hand` | 16 | 0 | 1 | 1 | mjINT_IMPLICITFAST | yes |
| `shadow_hand` | 20 | 4 | 1 | 1 | mjINT_EULER | yes |
| `wonik_allegro` | 16 | 0 | 1 | 1 | mjINT_EULER | yes |

## Required feature set

- `actuator_transmission:mjTRN_JOINT`
- `actuator_transmission:mjTRN_TENDON`
- `equality:mjEQ_WELD`
- `geom:mjGEOM_BOX`
- `geom:mjGEOM_CAPSULE`
- `geom:mjGEOM_CYLINDER`
- `geom:mjGEOM_MESH`
- `geom:mjGEOM_PLANE`
- `geom:mjGEOM_SPHERE`
- `integrator:mjINT_EULER`
- `integrator:mjINT_IMPLICITFAST`
- `joint:mjJNT_FREE`
- `joint:mjJNT_HINGE`
- `mocap_body`
- `solver:mjSOL_NEWTON`
- `tendon_transmission`

## Blocking requirements

These four are not negotiable, because dropping any of them changes what the
phase measures rather than how fast it measures it:

- **`tendon_transmission`** — `shadow_hand` drives 4 of its 20 actuators through
  tendons. A backend without tendon transmission cannot run Shadow, and the plan
  forbids resolving that by dropping Shadow from the gate.
- **`equality:mjEQ_WELD`** and **`mocap_body`** — every hand uses them: the
  `mocap-weld-v3` rollout protocol drives the wrist through a mocap body welded
  to the hand root. Without weld the acquisition protocol does not exist.
- **per-contact force and frame** — the multi-quantity safety budget is defined
  on resolved contact force, impulse and penetration. A backend that reports
  contact *existence* but not resolved force cannot carry the budget at all.

## Verdict

`unknown_pending_gpu_environment`

`mujoco_warp`, `mujoco.mjx` and `warp` are all absent from the CPU research
environment, and the repository pins no GPU lock for them. Support therefore
cannot be resolved here, and is recorded as unknown rather than assumed.

To resolve it, run this same script in an environment where `mujoco_warp` is
importable on a real NVIDIA device:

```bash
python scripts/phase3_4_backend_spike.py --out phase3_4_requirements.json
```

and compare `required_feature_set` against what that backend compiles.

## Decision rule

If the GPU backend cannot carry every blocking requirement, Phase 3.4 stays blocked and a backend decision record is written. Mocking CUDA or dropping Shadow from the gate is not an accepted resolution.

## SHA-256

- `requirement-matrix.json`: `42b2a0c2f77ca482061780e71e047bf94a8723aef076cb1e92d2b6f5c0cbde22`
