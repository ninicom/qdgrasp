# Kaggle T4 run v6 — independent reproduction, and a defect only a device finds

Kernel version 6, pinned at `af579cff2b0b93b0170062bd060aae5f9aff3144`.

## What it confirms

The compatibility matrix, run again on the `REV-20260827-010` V-003 reproducer:

| mujoco-warp | v5 errors | v6 errors |
| --- | --- | --- |
| 3.12.0 | 68224 | 65467 |
| 3.11.0 | 66181 | 67030 |
| 3.10.0.3 | 65248 | 69724 |

Counts differ run to run, which is what uninitialised memory does. The finding
does not: no version in this range is clean, on `warp-lang` 1.16.0, and 1.16.0
is the newest release there is.

`racecheck` stays at `0 hazards` while `initcheck` fires, which is what
separates an uninitialised read from a race.

## What it found in our own code

The capability probe passed this time -- the constraint force was located under
`efc.force` -- so the gate reached the performance benchmark at 1024 worlds on
the LEAP hand scene. It then died:

```
WorldRejected: peak_safety_metrics['max_object_speed_mps'] is not finite: nan
```

`RolloutSummary` refuses to hold a non-finite metric, which is right. The
producer was computing one anyway, so a single NaN world aborted the whole
rollout instead of being recorded as a rejected world. A NaN world is a
measurement -- `non_finite_state` -- not a crash.

On CPU no world ever goes non-finite, so no amount of local testing would have
surfaced this. Fixed in both backends after this run, with stub tests that
poison a world's velocity and require it to be rejected rather than raised on.

`cuda-gate.json` here therefore reads `NO_EVIDENCE`: the gate exited before
writing a verdict, and the notebook recorded that fact rather than leaving a
missing file to be discovered later.
