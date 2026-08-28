# S10 — run v13: admissible GPU evidence, and it still says FAIL

Tesla T4, commit `e79b1ea1e36a`. Gate verdict `FAIL`, and this time the verdict
is the only thing wrong with the bundle.

## What changed

The previous bundle failed the closure gate for two reasons that had nothing to
do with the GPU:

- it measured `9ab8e12`, and the candidate had moved on, so the evidence was no
  longer about the code under review
- it declared no `raw_log_sha256`, so its metrics could not be audited against
  the run that produced them — a requirement of WRK-R3 that the gate producing
  the bundles did not yet satisfy

Both are fixed. The gate writes `raw-run.log` beside the evidence and records its
digest; the closure verifier binds it. And the measured tree now agrees with the
candidate on the paths that matter:

    e79b1ea1e36a and b47d5015d8bf agree on
    ['qdgrasp', 'scripts/check_phase3_4_3_cuda.py']

So the distinction worth drawing: we no longer have *no valid GPU evidence*. We
have valid GPU evidence of failure.

## What the numbers say

| | leap_hand | wonik_allegro |
|---|---|---|
| speedup | 4.93x | 12.93x |
| non-finite worlds | **82 / 1024** | 0 |

Recomputed from the metrics, not read from the bundle:

    parity stage single_contact did not pass
    sanitizer reported errors
    sanitizer tool initcheck is not clean
    leap_hand: 82 world(s) rejected as non-finite

Same finding as v7 and v8, on a third independent sample. Speed passes on both
hands and cannot buy back correctness; section 10 asks for parity and a clean
sanitizer as well.

## What is no longer in this run

The MJX probe is gone. It answered its question once — `mujoco-mjx` installs and
`jax.jit(mjx.step)` does not finish compiling the contact-rich scene in the time
a probe should take, recorded in `kaggle-run-v11/` — and then cost four runs,
two of them by aborting the notebook before the gate below it ever ran. A
diagnostic that loses the measurement is worse than no diagnostic.
