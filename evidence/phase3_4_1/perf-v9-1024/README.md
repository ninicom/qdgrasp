# P3.4.1-01 stability triage: the divergence is classified

Two fresh runs at the pinned 1024-world operating point, LEAP scene, 91 geoms.

| | v9 | v10 |
| --- | --- | --- |
| speedup | 4.444x | 4.537x |
| non-finite worlds | 29/1024 | 34/1024 |

## What the diagnostics rule out

| check | value | conclusion |
| --- | --- | --- |
| `qpos_shape` vs expected | `[1024, 30]` vs `[1024, 30]` | not a state-reading defect in the benchmark |
| `diverged_are_contiguous_tail` | `False` | not an allocation or index-range error |
| rejected indices across runs | 29 then 34, scattered | not fixed by index |
| first diverged index | 4 | divergence starts early, not at a boundary |

Diverged indices in v10: `[4, 112, 150, 158, 210, 244, 256, 276, 277, 328, 361, 374, 385, 484, 522, 590, 647, 649, 652, 731]` (first 20).

## The finding

**`distinct_finite_qpos_rows = 990`.**

Every finite world holds a different `qpos`. All 1024 worlds are seeded from the
same `MjData` and stepped with identical commands, so they should be identical at
every tick. They are not: all 990 survivors differ.

This reframes the problem. It is not 34 bad worlds among 990 good ones -- it is
1024 worlds diverging from identical inputs, with the non-finite ones being the
extreme tail of that spread rather than isolated failures. A GPU-searched
candidate is therefore not reproducible even when it does not go NaN.

## Classification per the plan's decision tree

Section 3.4 offers five causes. The evidence selects one:

- *`Data.overflow` before NaN* -- unmeasured in these runs; the reader landed
  after v10 and needs one more run to confirm or exclude.
- *bad world fixed by index* -- **excluded**: indices are scattered and change
  between runs.
- *bad world changes between runs* -- **matches**: 29 then 34, different sets.
- *upstream MJWarp bug* -- still open; separating this from a wrapper defect
  needs `mjwarp-testspeed` on the exported model.
- *genuine solver instability* -- **unlikely to be the whole story**: real
  instability would not make 990 identical-input worlds each land somewhere
  different.

The plan's prescribed response for this class is a race, uninitialized memory or
atomic path defect, evidenced by Compute Sanitizer rather than by tuning.

## Next, in the plan's order

`racecheck` and `initcheck` on a minimized reproducer, and `Data.overflow` read
at the rollout boundary to close out the one remaining unexcluded cause.
