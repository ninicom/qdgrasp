# P3.1-13 controlled ablation evidence

Official selection evidence is `report-positive-control.json`, generated with
`--execute --matrix positive-control` from commit
`79a997b7ba2a568f1ea2f01812823cedb85ee0a4`. It contains 84/84 accounted
candidates and selects `region_opposition_v1` under the pre-registered
fail-closed rule.

SHA-256:

- `report-positive-control.json`:
  `f2831dc9db276c039c77c8503f4ec7e62cdc02b5f33e454ded9d425aead59186`

The other reports are retained as diagnostic history, not selection evidence:

- `report.json`: canonical procedural matrix; inconclusive because no recipe
  supplied three-hand dynamic evidence.
- `report-active-pair-coverage.json`: exploratory intervention; inconclusive and
  invalid for recipe selection because it used experimental solver semantics
  that were subsequently reverted after positive-control regression.

The positive-control result establishes recipe choice only for the validated
morphology-specific envelope. It does not establish procedural-object
generalization; the canonical failure remains an explicit limitation.
