# Kaggle T4 run v7 — the complete CUDA gate verdict

Kernel version 7, pinned at `7903f9bc3e47ecafdbac2be3171c5ac276ee7b7e`, which
the evidence records so it can be tied to a tree.

**Verdict: FAIL.** Not blocked, not skipped — the gate ran every stage and
several of them failed, which is a more useful result than either.

## What the device does well

| | |
| --- | --- |
| Capability | **supported** — every contact field readable, constraint force included |
| Parity, no contact | **pass** — max qpos delta `5.75e-10` against a `1e-4` tolerance |
| Parity, outcome class | **pass** — 4/4 worlds agree, every survivor exports a capsule |
| Speed, LEAP | **5.47x** CPU (13163 vs 2406 steps/s), criterion is `>=2x` |
| Speed, Allegro | **14.04x** CPU (25776 vs 1837 steps/s) |
| Device VRAM | 0.037 / 0.062 GiB against a 14 GiB budget |
| racecheck | `0 hazards displayed (0 errors, 0 warnings)` |
| Contact buffer | zero overflow on both hands |

The speed criterion the plan pins is met comfortably, on both active hands, on
the median of three runs each.

## What fails

| | |
| --- | --- |
| Parity, single contact | **fail** — object delta `8.39 mm` against a `2 mm` tolerance |
| Non-finite worlds | **84 of 1024** on LEAP; zero on Allegro |
| initcheck | uninitialised reads in `_linesearch_iterative_kernel` |

These are not four findings. They are one, seen four ways: the integrator agrees
with the CPU to ten significant figures while nothing is touching, and diverges
four times past tolerance the moment contact is involved; worlds die; and the
sanitizer names the kernel. The contact and constraint solver path reads
uninitialised memory.

## Why speed does not rescue it

`ROADMAP-P3.4-001` §10 asks for `>=2x` *and* zero non-finite worlds *and* zero
sanitizer errors. The backend is fast. It is not trustworthy, and a fast number
from an untrustworthy backend is not evidence of anything. No threshold was
adjusted to change this verdict.

## Disposition

Per `ROADMAP-P3.4.3-001` G07.5, with no clean `mujoco-warp` version available
and `warp-lang` 1.16.0 already the newest release, the remaining options are a
fallback backend that has passed capability and parity, or a blocked GPU gate.
The gate stays closed.
