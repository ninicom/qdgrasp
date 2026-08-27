# Reproducing this review packet

Everything below runs from a clean checkout of the pinned commit in
`packet.json`. `MANIFEST.sha256` lists every source and evidence file that
commit contains for Phase 3.4 and 3.4.1.

## On CPU

```bash
scripts/check_phase3_4.py --backend cpu --profile micro
python -m pytest tests/dynamic_grasp/ -q
scripts/phase3_4_1_shadow_audit.py
scripts/phase3_4_ablation.py
```

`check_phase3_4` reports `PARTIAL` by design and lists what is outstanding. It
should never report the phase as closed.

## On a real NVIDIA device

```bash
scripts/phase3_4_backend_spike.py --out requirements.json
scripts/phase3_4_cuda_contact_search.py --device cuda:0 \
  --profile kaggle-t4-micro --evidence cuda_evidence.json
```

The CUDA gate exits nonzero on a CPU host, on a non-CUDA device string, on any
non-finite world, on overflow, and when VRAM cannot be measured outside the
PyTorch allocator. All of those are intended.

The published runs used a Kaggle T4 kernel, MuJoCo Warp 1.16.0, torch
2.10.0+cu128. Run-to-run variation in the non-finite world count is itself one
of the findings, so an exact match on that number is not expected.

## What a reviewer should be able to reproduce, and what they should not

Reproducible: the CPU gate result, the test suite, the Shadow contact-pair
audit, the ablation verdict, the GPU speed figure at the pinned operating point.

Not reproducible by design: the exact set of diverged GPU world indices. That
set changed between runs (29 then 34, different indices), which is the evidence
for the non-determinism finding rather than a defect in the packet.

## Where the failures are

Failures are not separated out. `evidence/phase3_4/p16-dataset-blocked/` records
nine hand-iterations that produced nothing, `p14-ablation/` records a
`no_measured_difference` verdict, and `p15-throughput/` keeps the 0.764x figure
from the badly chosen 64-world operating point alongside the 4.444x that
replaced it.
