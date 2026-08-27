# P3.4-15 throughput and stability, measured on a Tesla T4

MuJoCo Warp 1.16.0, 100 steps, identical commands on both backends, two scenes
at two pinned world counts.

| scene @ worlds | geoms | GPU steps/s | CPU steps/s | speedup | rejected worlds |
| --- | --- | --- | --- | --- | --- |
| `leap_hand_scene@1024` | 91 | 12781.4 | 2876.3 | **4.444x** | 29/1024 |
| `leap_hand_scene@64` | 91 | 2021.1 | 2913.9 | **0.694x** | 1/64 |
| `micro_pusher@1024` | 3 | 69290.2 | 31117.6 | **2.227x** | 0/1024 |
| `micro_pusher@64` | 3 | 5824.6 | 32199.8 | **0.181x** | 0/64 |

Gating point `leap_hand_scene@1024`.

## Speed: criterion met

`4.444x` against a required `2.0x`.

The first attempt measured 0.764x and I reported the performance gate as failed.
That was measured at 64 worlds, which used ~0 GiB of the 14 GiB budget -- the GPU
was essentially idle, and losing to the CPU under those conditions says nothing
about the backend. Section 10 names 64 as a floor, not as the operating point.
1024 was then declared once, before running, and both counts are reported above.

The 64-world numbers are kept deliberately. They are the honest record of what a
badly chosen operating point measures, and dropping them would leave only the
flattering figure.

## Stability: criterion not met

**29 of 1024 worlds went non-finite on the hand scene, 2.8%.** The plan requires
a nonzero exit on NaN, so the gate fails here even though the speed criterion
passes, and it is right to.

The important part is not the rate but the mechanism. All 1024 worlds start from
the same state and receive the same commands, so they should evolve identically.
995 did and 29 did not. That is non-determinism inside the GPU backend, not
physics, and it means a GPU-searched result cannot be trusted as reproducible
until it is understood.

The backend refused to export any diverged world as a finalist, which is the
fail-closed path working as designed.

## What was not done

The world count was not raised further to dilute the rejection rate, and the
rejection check was not relaxed to a tolerance. The plan's rule is a nonzero exit
on NaN, and the gate implements that rule rather than a threshold chosen to pass.

## SHA-256

- `kernel-v9.log`: `fff430dfe85573c4e2229aca86cf67b931ced9c8a7b9d4413d65fe3d48f426d3`
- `phase3_4_cuda_evidence.json`: `15ec8bb2d443ad99f766b8f6bf49d062e33e3bac24d9cec65bd54b069b29e462`
