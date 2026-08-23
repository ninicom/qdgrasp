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
