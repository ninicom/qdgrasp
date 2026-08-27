# P3.4.1-07: option A measured, and it does not yet yield a passing Shadow

Plan section 4.3 option A -- keep the thumb/index active set, return MF/RF/LF to
a canonical open posture instead of 1.2/1.2/1.4 rad -- was applied at the recipe
source so every derived quantity (pinch vector, contact points, expected
fingertip positions) recomputes consistently, and then swept.

| inactive-finger value (rad) | validated rollout | peak normal force (N) | budget margin | failure stage |
| --- | --- | --- | --- | --- |
| 0.00 | fail | **0.6** | **+0.886** | `actuator_tracking` |
| 0.15 | fail | 3249.0 | -107.301 | `actuator_tracking` |
| 0.30 | fail | 6624.3 | -219.809 | `actuator_tracking` |
| 0.45 | fail | -- | -- | `actuator_saturation` |
| 0.60 | fail | -- | -- | `actuator_saturation` |
| 1.20 | fail | -- | -- | `actuator_saturation` |
| recipe as-is (1.2/1.2/1.4) | **pass** | 323.1 | -5.485 | -- |

## What this establishes

Option A solves the safety problem outright. At 0.0 rad the peak self-contact
force drops from 323 N to **0.6 N**, the budget margin goes from -5.485 to
**+0.886**, and damaging contacts go from 397 to **zero**, while the target is
still lifted 5.6 cm. Everything the grasp needs is intact: two active fingers
sustained, contact duty cycle `[1.0, 0, 0, 0, 1.0]` exactly as intended.

It also fails the validated rollout, on `actuator_tracking`. Intermediate
postures are far worse, not better -- 0.15 and 0.30 rad produce thousands of
newtons -- and anything from 0.45 rad up saturates the actuators.

So there is a real trade-off rather than a value left to tune: the posture that
is collision-free is one the actuator cannot hold, and the posture the actuator
holds comfortably is the one that self-collides.

## This reopens the classification

P3.4.1-06 classified the failure `invalid_posture` because a neutral posture is
clean. That reasoning is incomplete. The overlap at the recipe posture is
**0.36 mm**, and a real Shadow hand curls its fingers into the palm without harm;
318 N is the simulator's stiff contact response to a small geometric overlap, not
a force the hardware would see.

Whether that overlap is a posture defect or a collision-proxy defect cannot be
settled from contact numbers. Section 4.2 names the missing instrument: a visual
overlay of the collision proxy against the visual mesh at the first-contact
frame. Until that is run, `invalid_posture` and `invalid_proxy` are both live and
option A alone does not close SHADOW.

## State

The recipe source is **unmodified**. The change was applied, measured and
reverted, because shipping it would break the validated rollout and would
propagate a new `recipe_hash` into P2, P3.2 and P3.3 claims for a configuration
that does not pass.

No threshold, contact stiffness, actuator gain or force budget was altered. Plan
section 4.3 forbids those as a first fix and none was used.

## Addendum: the classification is settled, and the tracking failure is explained

Two measurements taken after the sweep close both open questions.

### `invalid_proxy` is excluded

| body | visual mesh half-extent | collision proxy |
| --- | --- | --- |
| `rh_lfproximal` | `[0.0101, 0.0100, 0.0309]` | capsule, radius `0.009`, half-length `0.02` |
| `rh_lfmetacarpal` | `[0.0126, 0.0214, 0.0368]` | box `[0.011, 0.012, 0.025]` |

The collision proxy is **smaller than the visual mesh on every axis of both
bodies**. It is conservative, not inflated. So if the proxies overlap by 0.36 mm
the visual geometry overlaps by more, and the contact is real rather than a proxy
artifact.

The recipe posture also sits well inside the joint limits -- `rh_LFJ1`/`LFJ2` at
76% of range, `LFJ3` at 91% -- so the model permits it and the links genuinely
intersect there. That excludes `missing_structural_exclusion` as well.

**`invalid_posture` from P3.4.1-06 stands.** The doubt raised earlier in this
document was wrong and is withdrawn.

### Why 0.0 rad fails actuator tracking

`rh_LFJ1` and `rh_LFJ2` have no individual actuators. One tendon actuator,
`rh_A_LFJ0` on tendon `rh_LFJ0` with ctrlrange `[0.000, 3.142]`, drives both;
the tendon coordinate is the sum. `rh_MFJ0` and `rh_RFJ0` are the same.

Commanding both joints to 0.0 therefore puts that tendon at the exact bottom of
its control range, which is the hardest point for a position actuator to hold.
That is a plausible mechanism for the tracking error and it has not been
confirmed further.

**Correction.** An earlier version of this section went on to claim the sweep was
an invalid parameterisation because it set coupled joints as if they were
independent. That is wrong. `FixedTendonTransmission`
(`qdgrasp/robot/transmission/fixed_tendon.py`) is exactly the component Phase 2
built for this, and it converts joint-space targets into the 20 actuator
commands correctly. The sweep's targets were valid and its results stand.

### Handoff

Option A remains the right direction and its safety result stands: 323 N to
0.6 N with damaging contacts eliminated, target still lifted 5.6 cm, duty cycle
`[1, 0, 0, 0, 1]`.

What is unresolved is narrower than previously written: a clearance posture is
needed that keeps the inactive fingers out of the palm **and** sits somewhere a
position actuator can hold. 0.0 satisfies the first and not the second; 0.15 and
0.30 satisfy neither. Whether such a posture exists in this hand's coupled
transmission is an open question, not a parameter left to pick.
