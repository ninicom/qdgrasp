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
