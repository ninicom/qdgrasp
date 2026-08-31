# Grasp Policy MVP evidence

Raw artifacts for `ROADMAP-MVP-001`, one directory per tune round.  `runs/` is
Git-ignored, so this tree is the auditable copy; each directory carries a
`MANIFEST.json` with the SHA-256 of every file in it.

| Directory | Round | Tier A | Tier B | Tier C | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `round-1/` | expert = best-margin residual | 37.0% | 34.7% | 23.5% | fail |
| `round-2/` | minimum-intervention expert + previous-action dropout | 16.0% | 15.7% | 10.0% | fail |
| `round-3/` | low-pass residual, predicate corrected to §4, noise-injected demos | 100.0% | 94.7% | 93.5% | pass |

Round 1 and round 2 keep their reports and ledgers but not their weight files:
the numbers and the per-episode ledgers are what a reader needs from a
superseded round, and the weights are reproducible from the committed scope,
prior and seeds.  Round 3 keeps `policy/bc.pt` (the rollback) and
`policy/ppo.pt` (the candidate).

Nothing here is release evidence.  The artifact is `experimental_non_release`,
its physics is MuJoCo CPU, and it has had no independent review.  Read
`docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md` before citing any number in it --
in particular the measured finding that the learned policy does not improve on
the controller prior.

Regenerate and re-check with:

```bash
python scripts/check_mvp.py --runs evidence/mvp/round-3
```
