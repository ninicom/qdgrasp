# P3.4-15 CUDA backend decision — measured on a Tesla T4

Resolves the question `P3.4-04` left open on the CPU research host: whether
MuJoCo Warp carries the four requirements the release models actually need.

Run on Kaggle, kernel `quyndang/qdgrasp-phase-3-4-cuda-gate` version 3, pinned at
the immutable commit recorded in the evidence JSON.

## Environment

| | |
| --- | --- |
| GPU | Tesla T4, capability 7.5, 14.56 GiB |
| torch | 2.10.0+cu128 (CUDA build 12.8) |
| MuJoCo | 3.12.0 |
| MuJoCo Warp | 1.16.0, devices ['cuda:0', 'cuda:1'] |

## Result

**Verdict: `supported`** — no unsupported requirement, no untested requirement.

| hand | tendons | `put_model` | contact readback |
| --- | --- | --- | --- |
| `leap_hand` | 0 | yes | yes |
| `shadow_hand` | 4 | yes | yes |
| `wonik_allegro` | 0 | yes | yes |

Shadow compiling with all four tendon actuators is the important line. It was
the largest risk in the plan: no tendon transmission would have meant dropping
Shadow or blocking the phase, and neither was needed.

## Correction carried in this evidence

Version 2 of this kernel reported `per_contact_force_and_frame: true` **without
testing it**. The `unsupported` list was only populated when `put_model` failed,
so any requirement the compile did not exercise defaulted to true. That was an
unearned claim in the gate whose whole purpose is to refuse unearned claims.

Version 3 makes capabilities tri-state — `supported` only when exercised,
`not_tested` otherwise — and actually probes the contact stream with `put_data`
plus twenty steps. The result above is measured, not defaulted.

## Precision of the contact claim

The probe confirms the per-contact stream is readable and exposes
`frame`, `pos`, `dist`, `geom` and `efc_address`. Force is not a direct field in
MuJoCo: it is resolved through `efc_address` into the constraint force array.
The handle is present, so the pathway exists, but P3.4-05 must dereference it and
verify the resulting magnitudes against the CPU oracle before any safety budget
number derived on GPU is trusted.

## What this does and does not establish

Unblocks `P3.4-05` (the MJWarp CUDA backend). It does **not** close Phase 3.4:
throughput, VRAM, CPU/GPU parity fixtures, a CPU-confirmed finalist per hand,
`QDGrasp-ContactRich-Tiny` and the independent review all remain outstanding.
The evidence records `search_benchmark: not_implemented` for
exactly that reason.

## Incidental finding

The same run is the first time `scripts/phase2_cuda_fk_parity.py` has ever
executed. It carried two dead calls — `require_cuda(device=...)` and
`environment_info(device=...)`, neither of which those functions accept — so the
Phase 2 CUDA FK parity gate had never run on any host despite Phase 2 being
recorded complete with that gate in its list. Both are fixed and the parity now
passes on real hardware.

## SHA-256

- `kernel-v3.log`: `ad76a683ef03db7dacaad9d2b8ca2b493993e46bb9611b6100f0eaa19a90618e`
- `phase3_4_cuda_evidence.json`: `ba36bbf4c17b60a366fcb29ba3ef49289956ff5e03b2ba8a816b19e41f1c4c8a`
