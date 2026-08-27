# P3.4-16 QDGrasp-ContactRich-Tiny — generated, release blocked

`release_blocked = True`. Reason: no measured dynamic positive for: shadow_hand.

9 samples, 4 positive,
5 negative. Hands with a measured dynamic positive:
leap_hand, wonik_allegro.

## Every sample

| hand | variant | outcome | lift (m) | peak normal force (N) | min budget margin |
| --- | --- | --- | --- | --- | --- |
| `leap_hand` | `baseline` | PASS | 0.0491 | 5.711 | 0.879 |
| `leap_hand` | `heavy_object` | PASS | 0.04848 | 5.84 | 0.8718 |
| `leap_hand` | `no_closure` | validated_rollout_failed | 2e-05 | 1.145 | 0.8408 |
| `wonik_allegro` | `baseline` | PASS | 0.04069 | 0.99 | 0.7502 |
| `wonik_allegro` | `heavy_object` | PASS | 0.03857 | 0.855 | 0.7489 |
| `wonik_allegro` | `no_closure` | validated_rollout_failed | 1e-05 | 0.123 | 0.76 |
| `shadow_hand` | `baseline` | damaging_contact | 0.05527 | 323.137 | -5.4854 |
| `shadow_hand` | `heavy_object` | damaging_contact | 0.03684 | 323.137 | -5.4854 |
| `shadow_hand` | `no_closure` | damaging_contact | 1e-05 | 323.137 | -5.4854 |

## What changed to get here

An earlier rollout reimplemented part of `mocap-weld-v3` and produced zero
positives across nine hand-iterations. Attaching the Phase 3.4 contact observer
to `validate_grasp_rollout` through its own `step_observer` hook -- reusing the
protocol Phase 3.2.1 certified rather than copying it -- produced real dynamic
positives immediately, with genuine `support_assisted` contact alongside the
target contact.

A second correction was needed. The safety budget judged **cumulative** normal
impulse over the whole rollout. Impulse is force times time, so that limit
rejects every sustained hold no matter how gentle: it measures how long a hand
held on, not whether holding was safe, and it made hands incomparable by grasp
duration. Impulse is now judged over a rolling window, which is what bounds an
impact; sustained load is bounded by peak force and `contact_duration_s`
instead. LEAP's margin went from negative to `+0.879` on the same physics.

## Why Shadow has no positive

`shadow_hand` lifts the target 5.5 cm and the validated rollout accepts it, but
the Phase 3.4 budget rejects it on measured self-contact. `rh_lfproximal`
against `rh_lfmetacarpal` reaches **323 N sustained** -- 28.5% of samples above
100 N, not a contact-onset transient. The release recipe closes LFJ1/2/3 to
1.2/1.2/1.4 while the grasp itself is a first-finger and thumb pinch, so the
little finger is driven into its own metacarpal.

This is a real finding rather than a tuning problem, and it is what the
multi-quantity budget exists to catch: Phase 3.2.1's criteria are about
hand-object contact and lift success, so they do not check self-contact load.

Leaving the little finger open was measured: peak force falls to 197 N and
damaging events halve, but other pairs still exceed the budget and the validated
rollout then fails. Tuning further would be selecting a control until the number
comes out right, so it was not done.

## What was deliberately not done

The budget was not raised for Shadow. The impulse correction above was made on
physics grounds -- the metric did not measure what it claimed -- and is applied
identically to all three hands. No threshold was moved to turn a rejection into
a pass.

## SHA-256

- `dataset_manifest.json`: `07bddaf34098d28ae43b332db6fb676b4553ae08817a95b9ccaa0296c4068826`
