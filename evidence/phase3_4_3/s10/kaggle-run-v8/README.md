# S10 — Kaggle run v8: the linesearch, isolated

Kernel version 8, Tesla T4, commit `9ab8e1204aa061c1063d7512198c59eb92c671b2`.
Gate verdict `FAIL`, matching v7. The new material is `solver_variants.json`.

## Why this run exists

v7 established that no `mujoco-warp` release in 3.10.0.3–3.12.0 is free of the
uninitialised read, and that `warp-lang` 1.16.0 is the newest release there is.
That closed the "pin a newer version" branch but left a cheaper one open: the
defect names `_linesearch_iterative_kernel`, and only some solver configurations
reach it. If one avoided the kernel, section 3.7 would accept it as a documented
fallback.

## Result — every configuration eliminated

| variant | status | initcheck |
|---|---|---|
| `baseline` | errors | 66 813 |
| `ls_iterations=1` | errors | **12 788** |
| `solver_cg` | errors | 850 170 |
| `ls_parallel` | **probe_did_not_run** | 0 errors — but see below |

`ls_parallel` printed `ERROR SUMMARY: 0 errors`, which is what a clean run prints.
It is not one. The probe raised before it ever stepped:

    AttributeError: ls_parallel was removed in MuJoCo Warp 3.9.1.

The positive-proof rule — `probe_stepped` *and* `0 errors`, never the absence of
an error line — caught it. This is the second run where that rule prevented a
false PASS; without it this evidence would read "ls_parallel is clean, the defect
is configurable away", which is the opposite of the truth. The parallel
linesearch is not an option to switch to: upstream deleted it.

## What the numbers say about the defect

`ls_iterations=1` cuts the count from 66 813 to 12 788 — the error count scales
with linesearch iterations, which corroborates the kernel attribution from
REV-20260827-010. It does not fall to zero, so the read happens on the *first*
iteration. There is no iteration count low enough to avoid it, because one is
already too many.

`solver_cg` is an order of magnitude worse, so switching solver families moves in
the wrong direction.

## Consequence

The configuration space is now exhausted by measurement, not by argument:

- no release in 3.10.0.3–3.12.0 is clean (v7)
- `warp-lang` 1.16.0 is the newest release (v7)
- `ls_parallel` no longer exists (v8)
- `ls_iterations=1` still leaks (v8)
- `solver_cg` is worse (v8)

Two paths remain, both new work needing their own plan: a different GPU backend
that passes capability and parity on its own evidence, or an upstream fix.
Section G07.5 permits the GPU gate to stay unpassed, which is where it stays.
