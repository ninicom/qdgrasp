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

## Addendum: the defect is upstream, in MuJoCo Warp's own solver

A rerun with `--print-limit 40` printed the records verbatim instead of counting
them. Every captured uninitialized read is in the same kernel:

```
Uninitialized __global__ memory read of size 4 bytes
  at _linesearch_iterative_kernel__locals__kernel_..._cuda_kernel_forward+0x7be0
  by thread (25,0,0) in block (0,0,0)
  Host Frame: _linesearch_iterative in solver.py:1359
  Host Frame: _linesearch           in solver.py:1562
  Host Frame: _solver_iteration     in solver.py:3533
  Host Frame: _solve                in solver.py:3726
```

Those frames are `mujoco_warp/_src/solver.py` -- MuJoCo Warp's own constraint
solver. QDGrasp appears in the backtrace only as the caller, at
`mjwarp_cuda.py:244` in `rollout`.

**This corrects the previous section of this document.** It said the next step
was auditing what `put_data` leaves untouched and whether
`MjWarpCudaBackend.reset` needs to initialize fields explicitly. That was a
reasonable guess and it is wrong: the read is in the solver's iterative
linesearch, not in per-world data initialization on our side. No change to
QDGrasp's reset path can fix it.

Section 3.4's row for this is *upstream MJWarp 1.16.0 bug*, whose prescribed
response is pinning a patch or a newer version through a compatibility spike,
with tendon, weld, contact, force and CPU parity re-evidenced.

### Limits

`racecheck` remains clean across both runs, so this is still not a race.

66,177 errors were reported and 40 printed; 3 records survived into the notebook
log, and all 3 are the same kernel. That is consistent with a single upstream
site but does not prove all 66,177 are, and a rerun capturing the full set would
settle it.

An `initcheck` report can also flag a value written by one kernel and read by
another in a way the tool cannot follow. The named frames make an upstream solver
defect the most direct reading, but confirming it means reproducing against
`mjwarp-testspeed` on the exported model, which section 3.3 step 6 already asks
for and which has not been run.

## Addendum 2: the wrapper is excluded; the defect is upstream, confirmed

Section 3.3 step 6. Every earlier run reached the solver through
`MjWarpCudaBackend`, so the backtrace named the kernel but never cleared the
wrapper. This run drives `put_model`, `put_data` and `step` directly with no
QDGrasp backend in the call path, on the same exported release model.

It reproduces identically:

```
upstream path stepped 8 times over 4 worlds, no QDGrasp backend
========= ERROR SUMMARY: 67531 errors
========= Uninitialized __global__ memory read of size 4 bytes
=========   at _linesearch_iterative_kernel__locals__kernel_..._cuda_kernel_forward+0x7be0
=========   by thread (25,0,0) in block (0,0,0)
```

Same kernel, same offset, same thread as the run that went through the backend.

### Determination

The root cause is an **uninitialized global memory read inside MuJoCo Warp
1.16.0's iterative linesearch**, in its own constraint solver. It is not a race:
`racecheck` reported zero hazards on every run. It is not a QDGrasp defect: the
wrapper is now excluded by construction rather than by inference.

Nothing in QDGrasp can fix it. Section 3.4's response for this row is a pinned
patch or a newer MJWarp version through a compatibility spike, with tendon, weld,
contact, force and CPU parity re-evidenced against whatever version is chosen.

### Consequences for the GPU gate

The speed criterion is met at 4.444x and 4.537x, but a GPU-searched result is not
reproducible while this holds: worlds seeded identically do not evolve
identically, and the non-finite ones are the tail of that spread. The gate's
nonzero exit on NaN is correct and should stay.

### What is still open

67,531 errors reported, 6 printed, and the captured records are all the same
kernel. That is consistent with one site and does not prove it is the only one.
Reporting this upstream, or bisecting MJWarp versions, would settle both that and
whether a fixed release exists.
