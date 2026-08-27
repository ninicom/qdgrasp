# P3.4.1-01 root cause: uninitialized memory, not a race

Compute Sanitizer on a minimized reproducer, Tesla T4, MuJoCo Warp 1.16.0.
`compute-sanitizer` was present at `/usr/local/cuda/bin/compute-sanitizer`.

| tool | result |
| --- | --- |
| `racecheck` | **`0 hazards displayed (0 errors, 0 warnings)`** |
| `initcheck` | **`ERROR SUMMARY: 62919 errors`** |

## The classification

Section 3.4's decision tree offered "race or uninitialized memory" for a bad-world
set that changes between runs. The two are now separated: racecheck is clean, so
this is **not a data race**. Initcheck reports 62,919 uninitialized-memory
accesses.

That explains the symptom exactly. The benchmark seeds every world from one
`MjData` and steps them with identical commands, and 990 of 990 survivors still
ended up different. If some `Data` fields are never initialized per world, then
the worlds do not in fact start from identical state -- each begins from
whatever was in device memory -- and "identical inputs" was never true.

## Supporting evidence from the horizon

| run | worlds | horizon | non-finite | distinct rows | max qpos spread |
| --- | --- | --- | --- | --- | --- |
| no sanitizer | 8 | 20 | 2 | 6 of 6 | **1.184** |
| under `initcheck` | 4 | 8 | 0 | 4 of 4 | 9.42e-05 |
| under `racecheck` | 4 | 8 | 0 | 4 of 4 | 4.63e-05 |

Under the sanitizer, execution is serialized and the horizon is shorter: the
spread is ~1e-4 and no world goes non-finite. Without it, over 20 steps, the
spread reaches 1.184 and worlds start failing. Small initial differences are
being amplified through contact dynamics until some worlds diverge to NaN.

Worlds remain distinct even at 1e-4 spread, so the divergence is present from
the start and the horizon only amplifies it. That is consistent with an
initialization defect and not with a solver instability that would affect all
worlds identically.

## What this points at

Section 3.4's prescribed response for this class is a reset, initialization and
indexing audit with every data field initialized. Concretely, the next step is
auditing what `mujoco_warp.put_data(model, cpu_data, nworld=N)` leaves untouched,
and whether `MjWarpCudaBackend.reset` needs to initialize fields explicitly
rather than relying on that call.

## Limits of this evidence

The notebook cell keeps only the last 4000 characters of sanitizer output, so
the individual error records were truncated and only the summaries survive. The
count and the racecheck verdict are solid; **which** arrays are read uninitialized
is not yet known and needs a rerun with `--print-limit` raised and the full log
captured.

Initcheck can also report an access as uninitialized when a value is written by
one kernel and read by another in a way it cannot follow. 62,919 is large enough
to be worth taking seriously, but the specific arrays still have to be named
before a fix is written.
