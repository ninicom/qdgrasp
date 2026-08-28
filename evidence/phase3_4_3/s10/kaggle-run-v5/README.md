# Kaggle T4 run — Phase 3.4.3 CUDA gate

Raw evidence from `kaggle.com/code/quyndang/qdgrasp-phase-3-4-3-cuda-gate-active-hands`,
kernel version 5, pinned at commit `144c585005a374b5c6b2daa37c2d47d67a86b318`.

## Environment

Tesla T4, compute 7.5, 14.562 GiB, driver 580.159.04, torch 2.10.0+cu128,
Python 3.12.13, `warp-lang` 1.16.0.

## What passed

- `phase1_cuda.json` — Phase 1 CUDA smoke: **PASS**.
- `phase2_cuda.json` — Phase 2 active-hand FK parity: **PASS**.
- racecheck on the reproducer: `0 hazards displayed (0 errors, 0 warnings)`.

## What did not

`warp_matrix.json` — the compatibility spike `§3.7` asks for, run on the
reproducer scene from `REV-20260827-010` V-003 (LEAP hand with its meshes),
under `compute-sanitizer --tool initcheck`:

| mujoco-warp | probe stepped | ERROR SUMMARY |
| --- | --- | --- |
| 3.12.0 | yes | 68224 errors |
| 3.11.0 | yes | 66181 errors |
| 3.10.0.3 | yes | 65248 errors |

Same kernel as the original record: `_linesearch_iterative_kernel`, thread
`(25,0,0)`, uninitialised `__global__` read of 4 bytes. racecheck stays clean,
which is what separates an uninitialised read from a race.

`warp-lang` 1.16.0 is the newest release there is, so there is no newer runtime
to move to either.

## Why the same matrix reported "clean" one run earlier

The previous run probed `tests/dynamic_grasp/micro_scene.xml` — three geoms —
and all three versions came back with `ERROR SUMMARY: 0 errors`. A toy scene
never reaches the kernel in question, so that result said nothing about the
defect. Both readings are kept here because the contrast is the point: a
compatibility answer is only as good as the scene it was asked on.

## Verdict

`cuda-gate.json` — **BLOCKED**. Per `ROADMAP-P3.4.3-001` G07.5 the remaining
options are a fallback backend that has passed capability and parity, or a
blocked GPU gate. Dropping the sanitizer, or filtering out the worlds it flags
after a rollout, are not among them.

The capability line in `cuda-gate.json` reads `missing_contact_fields:
["efc_force"]`. That was a defect in the probe, not in the build: MuJoCo Warp
moved the constraint force between `efc_force` and `efc.force`, and the probe
looked for one name. Fixed after this run; the capability answer here should be
read as unresolved rather than as negative.
