# P3.4-15 throughput: the performance gate fails, measured

Tesla T4, MuJoCo Warp 1.16.0, 64 worlds, 100 steps, identical commands on both
backends. Batch size is pinned at 64 rather than raised until it looks good.

| scene | geoms | GPU steps/s | CPU steps/s | speedup | rejected worlds | GPU warmup |
| --- | --- | --- | --- | --- | --- | --- |
| `leap_hand_scene` | 91 | 2364.8 | 3094.9 | **0.764x** | 1/64 | 0.834 s |
| `micro_pusher` | 3 | 6366.2 | 33438.2 | **0.19x** | 0/64 | 9.8723 s |

Gating scene `leap_hand_scene`; required `2.0x`,
measured **`0.764x`**, `speedup_met = False`.

## Result

**The performance criterion of plan section 10 is not met.** On the workload
Phase 3.4 actually searches -- a dexterous hand at 91 geoms -- the GPU backend
runs at 0.764x the CPU oracle: roughly 1.3 times *slower*, against a requirement
of 2 times faster.

The trend across the two scenes is the expected one. Three geoms gives 0.19x,
where per-step kernel launch dominates entirely; 91 geoms gives 0.764x. Batching
pays off as the model grows, and simply does not reach 2x at 64 worlds on a T4.

## What was not done about it

Raising the world count until the ratio crossed 2x. The plan pins the batch and
forbids looping upward precisely so a performance gate cannot be passed by
hunting for a configuration that passes it. The number above is the number at
the pinned configuration.

The gating scene was also not chosen after seeing results: both scenes are
reported, the rationale for gating on the hand is written into the script, and
the micro number is kept because it bounds the other end.

## Secondary finding

One world of 64 went non-finite on the hand scene and was rejected. The backend
refused to export it as a finalist, which is the fail-closed path working, but a
1-in-64 divergence rate on a release hand is worth understanding before any
GPU-searched result is trusted.

## SHA-256

- `phase3_4_cuda_evidence.json`: `db6524c5b681c1ca4aacb4a7957c4c9aa9f23b689ffef921d9229a7e02d0b34f`
