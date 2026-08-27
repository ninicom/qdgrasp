# IK iteration budget — measured negative finding

**Do not retry raising `max_iter` to fix canonical procedural yield. It was
measured and it does not work.**

## Why it looked promising

The P3.1-14 grid showed IK as the dominant failure stage across the whole
Allegro envelope (7-8 of every 16 candidates), and `run_pipeline_chunk` pins a
total budget of 80 iterations, split 40 + 40 across the two solver stages
(`qdgrasp/dataset/pipeline/orchestrator.py`, the `"max_iter": 40` solver kwarg
and the `max_iter=40` argument to `solve_joint_palm_dls_batch`). The obvious
hypothesis was that the solver runs out of budget just short of convergence.

## What was measured

`diag_ik_budget.py` records the **terminal residuals** of candidates on the four
canonical procedural objects, for all three hands, kinematics only. Twelve
hand x object cells, budget 8 candidates each.

Tolerances the solver must reach: position `0.001 m`, normal alignment
`dot >= 0.866` (30 degrees).

| | tolerance | best observed | typical median |
| --- | --- | --- | --- |
| position residual | 0.001 m | **0.0276 m** (28x) | 0.095 - 0.179 m (95 - 179x) |
| normal residual | 30 deg | **64 deg** | 86 - 151 deg |

Not one cell, on any hand, terminates anywhere near the tolerance. Fingertips
stop 3 - 18 cm away from their target contacts with normals pointing 60 - 150
degrees wrong.

## Conclusion

Damped least squares converges geometrically once it is inside a basin of
attraction. Terminating two orders of magnitude outside tolerance is not budget
exhaustion — the solver is not in a basin at all. Doubling or tripling the
iteration count cannot close a 100x gap.

The binding constraint is upstream: **the proposal stage hands the solver
contact sets that are not kinematically reachable for that hand.** Note also
that 1 - 6 of every 8 candidates are rejected at the proposal stage before IK
even runs.

Raising `max_iter` would have changed `generator_source_hashes`, invalidated the
P3.1-13 recipe-selection evidence and the P3.2.1 evidence chain, and cost a full
re-run — for no yield. That work was avoided by measuring first.

## Where to look instead

Canonical procedural yield is a **proposal feasibility** problem, not a solver
tuning problem. Any future attempt should target reachability-aware contact
proposal (rejecting unreachable contact sets before they are proposed), not the
solver budget.

## SHA-256

- `diag_ik_budget.py`: `284a5d9d6f8a35b5574eb3f4e29eb472d7a7e32db8a13f2421e0a8413aac8c39`
- `ik-terminal-residuals.json`: `b57d45499200b76bcdda8f9c3a4296a9d70c222b9f8ac7776c22f1427071f950`
