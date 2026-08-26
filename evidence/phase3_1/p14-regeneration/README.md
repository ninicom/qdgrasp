# P3.1-14 regeneration evidence

Raw evidence for the `DGN-Open-Tiny` regeneration that replaced the release
invalidated by `REV-20260823-009`.

## What each file is

- `canonical-baseline.log`: full regeneration on the twelve procedural objects
  with the selected recipe `region_opposition_v1`. **0 positives in all six
  shards.** This is the measurement that made the positive-control objects
  necessary; it is not a defect of the run.
- `variant-probe.json` / `variant_probe.py`: the pre-registered search for a
  second positive-control variant per hand. Sweeps were declared before being
  run: one parameter at a time, two values each, at the hand's validated
  candidate budget, then once more at the validated ceiling budget 16 for
  `wonik_allegro` only. No further values were run.
- `release-run1.log`, `release-run2.log`: the two clean regenerations. All six
  shards, all thirty-four object files and the complete manifest are
  byte-identical between them.

## Probe outcome

| hand | validated cell | second variant found |
| --- | --- | --- |
| `leap_hand` | positive | yes, `upper_height=0.055` |
| `shadow_hand` | positive | yes, `upper_center_z=0.135` |
| `wonik_allegro` | positive | **no** — 0/4 declared geometries, including at ceiling budget 16 |

The three validated positive-control cells reproduce outside the ablation
harness, which independently confirms `REV-20260825-001`.

`wonik_allegro` therefore holds a measured positive in one split only, its
`val` shard stays all-negative, and the generator keeps `release_blocked=True`.
The threshold, the release rule and the recipe were not changed to avoid that
outcome.

## SHA-256

- `canonical-baseline.log`: `ef2479a2519c2da7ca276d75fc81ecbf1c69a38e091aa562244e7030f9f1cc71`
- `release-run1.log`: `20e38711518839b3111093f52c08bd30c6644bd5db9a7693418cded59421a51b`
- `release-run2.log`: `d5427573b1ae88cfc209c66f8d3f81ecc65caee7a229c6a8021614283b5678e8`
- `variant-probe.json`: `1a560e97ec893928a9a0b57ed3bca669a4b44c44fb864075fdc2809e3b3e37cf`
- `variant_probe.py`: `bdaac1da728a982ac96f9716fac53a177d05a550999d68ab55e61a21d9ae6b36`
