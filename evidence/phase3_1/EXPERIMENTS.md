# P3.1 experiment ledger

Every experiment run against the P3.1 yield problem, including the ones that
failed. Read this before designing a new attempt: several plausible-looking
directions have already been measured and closed.

Outcome legend: `WORKED` adopted into the release, `DEAD` measured and rejected,
`DIAGNOSTIC` produced information but admitted nothing.

## Recipe selection

| # | Experiment | Outcome | Evidence |
| --- | --- | --- | --- |
| 1 | Controlled ablation, canonical procedural matrix, 3 recipes | `DEAD` — no recipe produced three-hand dynamic evidence; inconclusive | `p13-controlled-ablation/report.json` |
| 2 | Active-pair coverage intervention | `DEAD` — used experimental solver semantics later reverted; invalid for selection | `p13-controlled-ablation/report-active-pair-coverage.json` |
| 3 | Positive-control matrix, 84 candidates | `WORKED` — selected `region_opposition_v1` | `p13-controlled-ablation/report-positive-control.json` |

## Release yield (P3.1-14)

| # | Experiment | Outcome | Evidence |
| --- | --- | --- | --- |
| 4 | Full regeneration on the 12 procedural objects, selected recipe | `DEAD` — **0 positives in all six shards**. Procedural objects alone cannot produce a usable release | `p14-regeneration/canonical-baseline.log` |
| 5 | Reproduce the three validated positive-control cells outside the ablation harness | `WORKED` — 3/3 hands positive | `p14-regeneration/variant-probe.json` |
| 6 | Second variant via `upper_height` ∈ {0.045, 0.055} at validated budget | Partial — LEAP works at 0.055; Allegro and Shadow `DEAD` | `p14-regeneration/variant-probe.json` |
| 7 | Second variant via `upper_center_z` ± 5 mm at validated budget | Partial — Shadow works at 0.135; Allegro `DEAD` | `p14-regeneration/variant-probe.json` |
| 8 | Allegro: all four variants rerun at the fixture ceiling budget 16 | `DEAD` — still 0/4. Budget was not the constraint | `p14-regeneration/variant-probe.json` |
| 9 | Allegro kinematics grid, 5 widths × 5 block heights | `DIAGNOSTIC` — the floor-clearance diagnosis was **wrong**; IK dominates the whole envelope, and 40 mm is a poor operating point | `p14-regeneration/diag_allegro.json` |
| 10 | Allegro dynamic confirmation of the three best grid cells | `WORKED` — `width=0.045` measures 2 positives at both z=0.130 and z=0.140; `width=0.050, z=0.115` is `DEAD` | `p14-regeneration/confirm_allegro.json` |

Experiments 6-8 all held `width` at the pinned 40 mm. Width was the untried axis
and it was the decisive one. **A one-parameter sweep around a pinned calibration
point is not a search of the envelope.**

## Canonical procedural yield

| # | Experiment | Outcome | Evidence |
| --- | --- | --- | --- |
| 11 | Raise the IK iteration budget (`max_iter=40` per stage, 80 total) | `DEAD` — terminal residuals are 28-179x the position tolerance and 60-150 degrees off in normal. The solver is not in a basin; budget is not the constraint | `p16-ik-budget/` |

## Rejected without testing

These lower the bar rather than produce a grasp, and were never run:

- reducing object mass to make lifting easier;
- relaxing `min_palm_floor_clearance`;
- changing the `release_blocked` rule so an all-negative shard passes;
- loosening the position or normal tolerance;
- substituting a hand-built positive into the outcome list — this is exactly
  what `REV-20260823-009` invalidated.

## Open, not yet attempted

- **Reachability-aware contact proposal.** Canonical yield is a proposal
  feasibility problem: 1-6 of every 8 candidates die at the proposal stage, and
  those that survive hand the solver unreachable targets. This is where the
  canonical `0/12` should be attacked. Nothing here has been measured yet.
