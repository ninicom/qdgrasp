# Phase 3.2.1 failure corpora

Produced by `scripts/characterize_pipeline_failures.py`.  Each `corpus.json` is
a frozen record of what the real pipeline did on a pinned matrix — no repair, no
retry, no tuning.  Later remediation is credited only when it removes a specific
failure signature from this corpus while the unrelated signatures stay put.

## `baseline/` — control case (P3.2.1-01)

```bash
timeout 3600 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONHASHSEED=0 .venv/bin/python scripts/characterize_pipeline_failures.py \
  --label baseline --candidates 2
```

3 hands x 3 recipes x 2 candidates on the pinned 5 cm box, seed 42:

| Metric | Value |
| --- | --- |
| candidates | 18 |
| failure signature | `ik:max_iter` x 18 |
| candidates reaching dynamic rollout | 0 |
| pipeline-generated positives | 0 |

This reproduces the 18/18 `IK: max_iter` diagnostic that opened P3.2.1.

### What the residual telemetry shows

Every candidate burns the full 80 iterations, but the two residual families
behave very differently:

| Hand | Recipe | max position residual (m) | max normal residual (rad) |
| --- | --- | --- | --- |
| leap_hand | surface_fixed_v1 | 0.0197 | 2.267 |
| leap_hand | region_opposition_v1 | 0.0077 | 2.211 |
| leap_hand | wrench_guided_v1 | 0.0104 | 1.536 |
| wonik_allegro | surface_fixed_v1 | 0.0020 | 2.239 |
| wonik_allegro | region_opposition_v1 | 0.0059 | 2.366 |
| wonik_allegro | wrench_guided_v1 | 0.0098 | 2.606 |
| shadow_hand | surface_fixed_v1 | 0.0119 | 2.413 |
| shadow_hand | region_opposition_v1 | 0.0024 | 2.192 |
| shadow_hand | wrench_guided_v1 | 0.0028 | 2.129 |

Position error lands in the millimetre range while the contact-normal error sits
at 1.5-2.6 rad, i.e. the fingertips arrive near the target points pointing the
wrong way entirely.  That is the signature RC-01 predicts (the residual scores
the configured contact axis while the autodiff Jacobian differentiates a
parent-to-tip vector, so the normal term is never actually descended) and it is
the signature an RC-01 intervention must move.

Two further observations recorded but **not** yet attributed to a root cause:

- `wonik_allegro` palm hypotheses land at negative z (e.g. -0.069 m), i.e. below
  the floor plane — consistent with H-01/RC-03, to be tested in P3.2.1-06.
- no candidate reaches collision admission, so no collision, static or dynamic
  signature exists in this corpus yet; those stages are uncharacterized until
  the IK stage stops absorbing every candidate.

## `baseline-repeat/` — determinism control

Identical invocation with label `baseline-repeat`.  Stage accounting, failure
signatures and per-candidate telemetry match `baseline/` exactly (verified
field-by-field, excluding timestamp and wall-clock duration).

## `p02-refactor-parity/` — behaviour-preserving extraction (P3.2.1-02)

Both DLS solvers now call `qdgrasp/dataset/pipeline/contact_state.py` instead of
carrying private copies of the contact primitives.  The extraction keeps the
divergent `parent_to_tip` autodiff convention on purpose, so the corpus must not
move — and it does not: `changed_cells: 0`, and per-candidate telemetry compares
equal field by field against `baseline/`.  Anything that moves from here on is
attributable to the intervention that moved it.

## Findings recorded but not yet acted on

### H-04 — the normal task is graded far harder than it is descended

`normal_weight = 0.01` scales the direction block of both the residual and the
Jacobian, so it enters the normal equations at `1e-4` relative to the position
block, while convergence tests the raw dot product against `cos(30 deg)`.  The
corpus shows the consequence directly: `wonik_allegro/surface_fixed_v1` (0.0020 m),
`shadow_hand/region_opposition_v1` (0.0024 m) and `shadow_hand/wrench_guided_v1`
(0.0028 m) all satisfy `pos_tolerance = 0.005` and still return `max_iter` — only
the normal criterion is holding them back.  Per the plan's section 4, no weight
is retuned before the oracle exists; this is filed as a hypothesis to test after
RC-01, not a fix to apply now.

### RC-01 is hand-specific

Measured angle between the graded direction (configured contact axis) and the
differentiated one (parent-to-tip), averaged over interior joint states:

| Hand | divergence |
| --- | --- |
| leap_hand | 18.1 deg |
| wonik_allegro | 0.0 deg |
| shadow_hand | 22.0 deg |

Allegro's configured axis coincides with its parent-to-tip vector, so the RC-01
intervention is predicted to move the leap and shadow cells and to leave the
three Allegro cells unchanged.  Allegro's 2.2-2.6 rad normal residuals therefore
need a different explanation — H-04 and the palm hypothesis (H-01 / RC-03) are
the open candidates.  This prediction is pinned in
`tests/test_contact_state.py::test_rc01_divergence_is_hand_specific`.

## `p03a-rc01-direction/` — RC-01 intervention (P3.2.1-03)

Single variable changed: `AUTODIFF_DIRECTION_MODE` flipped from `parent_to_tip`
to `configured`, so the Jacobian differentiates the same direction the residual
is graded against.  Expected signature was recorded before the run: the LEAP and
Shadow cells move, the three Allegro cells do not.

Observed, on maximum normal residual (rad) per candidate:

| Hand | candidates | mean change | best change |
| --- | --- | --- | --- |
| leap_hand | 6 | -0.092 | -0.194 |
| shadow_hand | 6 | -0.082 | -0.194 |
| wonik_allegro | 6 | 0.000 | 0.000 |

The Allegro negative control holds exactly — its telemetry is unchanged to every
recorded digit, as the 0 deg divergence predicts.  LEAP and Shadow normal
residuals improve, and position residuals stay where they were.

**Verdict: RC-01 supported, and insufficient on its own.**  The mechanism is
confirmed — the predicted cells moved, the predicted control did not — but the
effect is roughly 0.1 rad against residuals of 2.1-2.7 rad, and the stage
signature is still `ik:max_iter` x 18.  Nothing here licenses closing the IK
failure; the remaining gap belongs to H-04 and the palm hypotheses.

## `p03b-rc02-mask/` — RC-02 intervention (P3.2.1-03)

Single variable changed: the active mask is applied to the Jacobian rows when
assembling the normal equations, not only to the error vector.

Corpus unchanged, exactly as predicted and recorded beforehand: the orchestrator
never passes `active_fingers`, so every finger is active and the mask is the
identity.  This is **not** evidence against RC-02.  The bug is in the curvature,
and target perturbation cannot detect it either, since the error vector was
already masked.  RC-02 is validated instead by
`tests/test_active_mask_invariance.py::test_inactive_jacobian_rows_do_not_enter_the_normal_equations`,
which scales inactive Jacobian rows by 1e6 and requires `H` and `g` to be
untouched, and by the reduced-system equality test beside it.

The corpus can only exercise RC-02 once P3.2.1-05 makes proposals emit a real
active set instead of forcing every fingertip active.

### H-05 — the RC-01 fix flips one hand-built LEAP fixture

Correcting the Jacobian changes which null-space posture the solver settles into,
and for the LEAP pinch fixture (`tests/test_physics_rollout.py`,
`tests/test_phase3_2_dynamic_fixtures.py`) that posture no longer holds the box
with two fingers.  Measured per-finger contact force at the end of the rollout,
object weight 0.196 N:

| Autodiff direction | verdict | per-finger contact force (N) |
| --- | --- | --- |
| `parent_to_tip` (pre-fix) | pass | 3.62, 0.00, 0.00, 3.60 |
| `configured` (post-fix) | fail | 0.00, 0.00, 0.00, 2.37 |

This is not a threshold artifact — the thumb loses contact outright, not by a
millinewton.  The task the fixture poses (hold the achieved contact directions
while pushing 3 mm along them) does not constrain the posture that decides
whether the thumb stays on the face, so the fixture's verdict was riding on a
degree of freedom nobody specified.

Checked against the other two hands: the Shadow fixture (22 deg divergence, the
same class as LEAP's 18 deg) still passes, and the Allegro fixture is unchanged
as its 0 deg divergence requires.  So this is one fragile fixture, not a
systematic loss of grasp quality.

Handling, per the plan's section 4 (no tuning before the oracle exists):

- the two tests are `xfail(strict=True)` naming this finding, so they report
  again the moment the verdict moves;
- `scripts/check_phase3_2.py` is left **red** on its LEAP fixture rather than
  given an allowlist — it is the regression detector for P3.2.1-04 through -08,
  and a gate that carries exceptions stops being one;
- nothing in `contact_load.py` or the rollout protocol is touched here.  The
  single-frame `final_active_fingers` count against an uncalibrated 1 mN
  threshold is a predicate the plan's section 3.6 already condemns; replacing it
  with a windowed sustained-contact measure against a calibrated noise floor is
  P3.2.1-09.

## `p04-solver-progress/` — measured failure classification (P3.2.1-04)

The solver now records task cost, accepted/rejected steps, raw/projected step
norms, joint-limit clipping, gradient norm, masked-Jacobian rank/condition,
final damping and finite-state status.  Failure reasons are classified from
those signals without changing geometry, tolerances, candidate budget or
iteration budget.

The frozen 3 hand x 3 recipe x 2 candidate matrix remains 18/18
`ik:max_iter`, reaches no dynamic rollout and produces no positive.  This is a
meaningful result rather than a failed intervention: every candidate retained
finite, non-zero-rank Jacobians and accepted measurable descent during its
fixed 80-iteration budget, so `singular`, `joint_limit`, `line_search_failed`
and `stagnation` would be false diagnoses.

Across the 18 candidates:

- accepted steps range from 5 to 38;
- rejected steps range from 42 to 75;
- final damping reaches the pinned cap `1.0` for every candidate;
- masked Jacobian rank is 16 for LEAP/Allegro and 22 for Shadow;
- task cost decreases, but the geometric convergence predicates remain unmet.

This isolates the next work: proposal active-set semantics and constrained palm
hypotheses.  It does not license increasing `max_iter`, retuning
`normal_weight`, or weakening convergence tolerances.

## `p05-active-opposition/` — explicit proposal task identity (P3.2.1-05)

At this historical intervention, region and wrench-guided proposals carried a
stable content ID, a three-contact active set (thumb plus two opposing fingers),
and opposition-pair identity.  Active opposing anchors were distinct and spatially separated without
assuming that a coarse planar surface contains an arbitrary number of triangle
IDs.  Surface-fixed remains the all-finger control recipe.

The orchestrator applies the same mask to palm fitting, IK, post-IK surface
admission, static certification and open/preload command IK.  The frozen corpus
shows the expected discriminating behavior:

- all six surface-fixed candidates are byte-identical in residual telemetry to
  `p04-solver-progress`;
- region/wrench candidates carry three active fingers;
- their masked task Jacobian rank falls from 16 to 12 for LEAP/Allegro and from
  22 to 13-14 for Shadow, proving the mask reaches the solver curvature;
- stage accounting remains 18/18 `ik:max_iter`, with no dynamic rollout and no
  positive.

P10 later tightened the production task to one thumb plus one enumerated
non-thumb finger.  That correction matches the two-opposition-group contract,
keeps static/dynamic contact count explicit, and avoids treating an arbitrary
third fingertip as part of the kinematic task.  The P05 corpus remains valid as
historical evidence for mask propagation, not as the current active-set shape.

P3.2.1-05 therefore closes proposal task identity but does not claim a yield
fix.  The remaining large position/normal residuals now belong to the
unconstrained palm initializer addressed by P3.2.1-06.

## P3.2.1-06 — constrained palm hypotheses and bounded local refinement

The initializer now enumerates explicit opposition/gravity grasp frames across
the pinned joint seeds, keeps Kabsch as a separately measured fallback, applies
a hard palm-floor admission check, and performs one bounded local SE(3)
correction between two 40-step joint-only solves.  The total 80-step IK budget,
all convergence thresholds and all task weights remain unchanged.

`p06a-grasp-frame/` and `p06b-local-se3/` preserve valid stage and residual
observations, but their recorded hypothesis ID/initial metrics are invalid for
mode-level analysis: an orchestrator bookkeeping bug stored the final loop
variable instead of the hypothesis in the selected `best` tuple.  The selected
pose itself was correct.  The bug is fixed before `p06c-direction-fit/`; no
mode-level conclusion below uses the two earlier records.

P06c adds an independent, unweighted direction Procrustes hypothesis.  Its
direction covariance must have rank at least two; rank-one fits are rejected
because palm roll is unobservable and the SVD result is not world-frame
equivariant.  The pose, permutation, floor, trust-region and world rigid-transform
oracles pass.

Against P06b, with the active mask applied when aggregating residuals:

- mean initial minimum normal alignment moves from `-0.477` to `0.001`;
- mean active maximum normal residual improves from `1.582` to `1.485 rad`
  (about 6.1%);
- mean active maximum position residual worsens from `8.89` to `13.09 mm`;
- the stage signature remains 18/18 `ik:max_iter`, with zero downstream and
  zero generated positives.

**Verdict: H-01 is rejected as the primary explanation of the 18/18 IK
failure.**  A constrained initializer changes the predicted orientation
telemetry, but does not remove the failure signature under the controlled
budget.  P06 closes the initializer contract; it does not claim to close IK.

## `p07-exact-collision/` — compiled MuJoCo admission (P3.2.1-07)

The trimesh probe remains a prefilter.  Final collision admission now compiles
the same hand XML and procedural object geoms as rollout, places the full
articulated state at the requested world palm pose, and records named geom/body
pairs, signed contact distance, maximum penetration and minimum hand-floor
clearance.  Only declared active fingertip bodies may contact the target object;
inactive tips, palm/non-tip links, excessive active-tip penetration, hand-floor
contact and hand self-collision fail closed.

Mutation tests prove the exact predicates independently: active-tip/object
contact passes; the same contact with an inactive mask fails; palm penetration
fails with geom evidence; and floor penetration fails independently of object
contact.  The source provenance manifest now includes proposal identity, palm
hypotheses and exact collision admission.

The frozen matrix remains byte-equivalent in stage accounting to P06c:
18/18 candidates stop at IK, so the exact collision stage is not reached and no
positive is produced.  This is the expected negative control, not evidence that
collision admission creates yield.

## `p08-task-command/` — controllable active-task command (P3.2.1-08)

The command layer no longer decides a contact task from the global joint-space
null-space norm.  It solves the active fingertip displacement directly with
`dq` parameterized in `range(M.T)`, preserving controllability while applying a
global step/joint-limit scale.  It records task residual and global null-space
residual separately, and rejects actuator saturation before a clipped command
can enter `mj_step`.

The rollout computes the point Jacobian from the compiled MuJoCo state and the
physical hinge/slide joints, rather than reusing the independent-joint FK
dimension.  The Shadow integration oracle therefore measures the real
`24 joint-state x 20 control` mapping.  Component tests cover:

- a task with a controllable equivalent despite an irrelevant global nullspace;
- a task lying entirely in the transmission nullspace;
- control-range saturation;
- a broken moment-matrix mapping that flips the verdict;
- Shadow's compiled 24-by-20 task path.

The P08 frozen matrix is unchanged from P07: 18/18 `ik:max_iter`, zero dynamic
entries and zero positives.  P08 closes command admission at component level;
it does not claim full-flow validation because the canonical corpus never
reaches this stage.

## `p09-dynamic-predicate/` — measured, fail-closed rollout verdict (P3.2.1-09)

The dynamic verdict is now a named conjunction over simulation stability,
actuator and palm tracking, sustained active contact, absence of palm/floor
support, penetration, lift, disturbance survival and friction-cone evidence.
Contact duty cycle and normal impulse are measured over pinned windows against a
noise floor calibrated in the same compiled scene.  Scenario/category names do
not enter the verdict.

Mutation tests independently flip every predicate and require the exact failure
stage.  Protocol admission also rejects invalid/non-finite thresholds and a
compiled hand with zero actuated damping before the first `mj_step`.  The P09
frozen corpus still records 18/18 IK rejections because it predates the P10
reachability interventions; it is evidence for predicate/source provenance,
not full-flow yield.

## P3.2.1-10 — generated-reachable full-flow positives

`generated_reachable.py` exposes only mesh, exact collision geoms, mass, object
pose and candidate budget.  It contains no q/qpos, joint target, palm pose,
contact point or accepted grasp.  The production pipeline discovers the full
provenance chain under a 16-candidate budget:

| Hand | Accepted | Lift | Max penetration | Sustained active contacts |
| --- | ---: | ---: | ---: | ---: |
| LEAP | 1 | 48.60 mm | 0.280 mm | 2 |
| Allegro | 1 | 49.37 mm | 1.431 mm | 2 |
| Shadow | 1 | 49.84 mm | 0.667 mm | 2 |

The causal Shadow diagnosis was not “increase iterations.”  A 43 mm pinch
produced adjacent-finger self-collision; deterministic opposition-finger
enumeration and active-mask-consistent morphology seeds exposed a collision-free
50 mm little-finger/thumb candidate.  The same generated grasp lifted 5 g but
correctly remained floor-supported at 20 g, establishing the pinned positive
load envelope.  Static certification was also aligned with the rollout's
`condim=4` soft-finger torsional friction model.

## P3.2.1-11 — canonical-independent matrix

The gate runs the same `region_opposition_v1` full path over four pinned,
hand-independent objects (50 mm box, 50 mm cylinder, 50 mm superquadric and
compound T) for all three hands, eight candidates per cell.  All 12 cells are
present in `canonical/manifest.json`; none produces a positive.  Failures remain
dominated by unavailable palm hypotheses and `IK:max_iter`, with two compound
cells reaching exact collision rejection.

This `0/12` is retained as a measured generalization limitation.  It does not
invalidate the narrower P10 existence claim, and P10 positives are not inserted
into canonical outcomes.

## P3.2.1-12/P3.2.1-13 — mutation and deterministic regeneration gates

`scripts/check_phase3_2_1.py` runs 100 contract/mutation tests, requires one full
generated positive for every hand, records the 12-cell canonical matrix, and
writes two independent generated staging manifests.  The manifests include
source, robot profile, object and rollout-protocol hashes plus raw stage and
trajectory evidence.

- `run-a/manifest.json` and `run-b/manifest.json` are byte-identical. Their
  canonical JSON payload SHA-256 is
  `5a34f9d8e7f6568c7dc28e1b5f70c0421b1910ef9fe038d75e33efec55b3c563`;
  the newline-terminated file SHA-256 is
  `9fcc4ddaa1e873c8a74de03ab2d8f2fadba2a44e4067ccb09cbb76bdd73636d5`.
- `canonical/manifest.json` payload SHA-256 is
  `20450c16e2bc74ca6079a7407b9d2573c05784335dcdac3c88f0a6531c3a8eaa`;
  its newline-terminated file SHA-256 is
  `03ef765a28f4f20330c7881e6e2d0f4a58bf8077f9655c83b62b9c1ead4d3116`.
- Full regression result: `366 passed, 1 skipped`, with no xfail.

## P3.2.1-14 — independent review and release decision

The first independent read-only review found one definite P2 issue: a task
displacement with zero active fingers reached `np.concatenate([])` instead of
failing closed.  The rollout now rejects both zero and one active finger with
`insufficient_active_fingers` before `mj_step`, and the release gate contains
both regressions.  A second independent pass found no definite correctness
issue.  `TPR-20260824-001` records the review and `REV-20260824-001` records the
narrow release decision.

P00–P14 are complete.  This establishes generated-reachable existence for all
three hands and closes the P3.2.1 correctness contracts.  The independent
canonical matrix remains `0/12`; canonical yield/generalization and the rest of
Phase 3 remain open work.
