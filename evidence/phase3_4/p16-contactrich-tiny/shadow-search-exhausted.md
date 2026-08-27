# Shadow has no dynamic positive, and it is not a search problem

Four independent lines of evidence, all measured, all pointing the same way.

| attempt | result |
| --- | --- |
| Wrapped rollout, release recipe as-is | `rh_lfproximal`/`rh_lfmetacarpal` at 323 N sustained, 28.5% of samples above 100 N -> `damaging_contact` |
| Wrapped rollout, `no_closure` variant | still `damaging_contact` with no finger closing at all |
| Little finger held open | peak force falls to 197 N, still over budget, and the validated rollout then fails |
| **CEM, 40 candidates over per-finger closure scale in [0,1]^5** | **40/40 scored -inf, every one hard-rejected** |

The CEM run is the decisive one. P3.4-09 exists precisely to search control
parameters, the space was declared before the run, the budget and objective were
untouched, and it found nothing. Elite closure scales converged toward low values
(ring finger to 0.1) and still every candidate violated the budget.

## What this means

Shadow's rejection is not something a control search can remove. The violation
survives with no closure at all, which places it in the pregrasp configuration
and the hand's own collision geometry rather than in the grasp. Fixing it means
revisiting Shadow's release recipe or its collision model, which is Phase 3.2 and
3.3 territory, not Phase 3.4 search.

Phase 3.2.1 does not catch it because its acceptance criteria are hand-object
contact and lift success. Self-contact load is not among them. That is the gap
the Phase 3.4 multi-quantity budget was built to expose, and here it exposed one.

## What was not done

The Shadow budget was not raised. The one budget change made this session --
windowing impulse instead of accumulating it over the whole rollout -- was made
on physics grounds, because a cumulative impulse limit rejects every sustained
hold regardless of safety, and it was applied identically to all three hands.
LEAP and Allegro pass under exactly the same budget Shadow fails.

## Discarded probe

An earlier probe set Shadow's joint values without positioning the hand root and
reported a 45,987 N contact with the floor. That measured a setup error in the
probe -- the hand sat at the world origin, 6.75 cm inside the floor -- and is not
evidence about the recipe. It is recorded here so nobody rediscovers it and
mistakes it for a finding.
