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
