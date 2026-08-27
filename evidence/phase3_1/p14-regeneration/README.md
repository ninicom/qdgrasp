# P3.1-14 regeneration evidence

Raw evidence for the `DGN-Open-Tiny` regeneration that replaced the release
invalidated by `REV-20260823-009`.

## Sequence

1. `canonical-baseline.log` — full regeneration on the twelve procedural objects
   with the selected recipe `region_opposition_v1`. **0 positives in all six
   shards.** This is what made positive-control objects necessary; it is also
   the standing limitation on procedural generalization.
2. `variant-probe.json` / `variant_probe.py` — pre-registered search for a
   second positive-control variant per hand. One parameter at a time, two
   values each, at the hand's validated candidate budget, then once more at the
   validated ceiling budget 16 for `wonik_allegro`. LEAP and Shadow each found
   a variant; `wonik_allegro` measured **0/4**. All four held `width` at the
   pinned 40 mm.
3. `release-run1.log` / `release-run2.log` — the resulting release: measured
   positives in five of six shards, `release_blocked` still true.
4. `diag_allegro.py` / `diag_allegro.json` — kinematics-only grid (5 widths x 5
   block heights, budget 16, no physics) characterizing **where** Allegro
   candidates die. It does not admit anything.
5. `confirm_allegro.py` / `confirm_allegro.json` — bounded dynamic confirmation
   of the three best grid cells, declared before running.
6. `release-final-run1.log` / `release-final-run2.log` — the accepted release.
   All six shards carry a measured positive, `release_blocked=false`, and the
   two runs are byte-identical across shards, objects and manifest.

## What the grid showed

The blocker was misdiagnosed as palm floor clearance. `palm_hypothesis_unavailable`
dominates only near Allegro's calibrated point (low block height at 35-40 mm
width). Across the rest of the envelope the binding constraint is **IK
convergence** (`max_iter`, hardcoded to 40 in the orchestrator): 7-8 of every 16
candidates die there even in the best cells.

The calibrated 40 mm opposition is a poor operating point for Allegro. Widening
to 45 mm measures two dynamic positives, against one for the calibrated fixture:

| width | block height | dynamic positive |
| --- | --- | --- |
| 0.045 | 0.140 | 2 |
| 0.045 | 0.130 | 2 |
| 0.050 | 0.115 | 0 |

`pc_allegro_02` uses `width=0.045, upper_center_z=0.130`.

## What was not done

No threshold, recipe, release rule, object mass or floor-clearance limit was
changed. Every positive in the release is a label the pipeline produced.
Procedural-object yield remains `0` and no generalization is claimed.

## SHA-256

- `canonical-baseline.log`: `ef2479a2519c2da7ca276d75fc81ecbf1c69a38e091aa562244e7030f9f1cc71`
- `confirm_allegro.json`: `03d75297339a0f6dfd2c4dc085599b87fc65c1ca518faddea9ed831fc51b9116`
- `confirm_allegro.py`: `a3f6dc51b1577a74685da39d128fad3230110d2a1d2419929c2b1872b1c0c62b`
- `diag_allegro.json`: `7607da918bce2dd0af7b8f292f419b7a0e0666151508b79d67bf7dd4f83fd6ae`
- `diag_allegro.py`: `8e0fc533d930da7b7a882b593fc18f0631a709c2a29c60ed93cd4a4263861b25`
- `release-final-run1.log`: `28dec37494ae512f4e09b8ec10cd2b8a146a2e7c0ba2f11ff6924b53ed0a9c30`
- `release-final-run2.log`: `7b7868b7085925b8b78540a25bd6605ed30cbb15cc7548ec0181acd75d660323`
- `release-run1.log`: `20e38711518839b3111093f52c08bd30c6644bd5db9a7693418cded59421a51b`
- `release-run2.log`: `d5427573b1ae88cfc209c66f8d3f81ecc65caee7a229c6a8021614283b5678e8`
- `variant-probe.json`: `1a560e97ec893928a9a0b57ed3bca669a4b44c44fb864075fdc2809e3b3e37cf`
- `variant_probe.py`: `bdaac1da728a982ac96f9716fac53a177d05a550999d68ab55e61a21d9ae6b36`
