# P3.4.1-06 Shadow localization: `invalid_posture`

Gravity and control are off in every stage, so any contact below is produced by
the commanded joint vector alone -- not by a controller pressing.

| posture | self-contacts | worst pair | penetration (mm) | normal force (N) |
| --- | --- | --- | --- | --- |
| `q_neutral_all_zero` | 0 | -- | 0.000 | 0.0 |
| `q_inactive_open` | 0 | -- | 0.000 | 0.0 |
| `q_lf_zero` | 2 | rh_rfproximal / rh_palm | 0.150 | 199.8 |
| `q_recipe_pregrasp` | 3 | rh_lfproximal / rh_lfmetacarpal | 0.360 | 318.3 |
| `q_recipe_closed` | 3 | rh_lfproximal / rh_lfmetacarpal | 0.360 | 318.3 |
| `q_closed_active_only` | 3 | rh_lfproximal / rh_lfmetacarpal | 0.360 | 318.3 |

## Classification

**`invalid_posture`.** A neutral posture is clean and opening every inactive finger clears the penetration entirely, while the recipe's values reproduce it. The recipe commands fingers the pinch does not use into the palm; the collision geometry is sound.

The neutral all-zero posture is completely clean: zero self-contacts, zero
penetration. So the Shadow collision geometry is sound and this is not
`invalid_proxy`. The pair is also not parent-child, so it is not
`missing_structural_exclusion`. And it is not `legitimate_self_contact`, because
opening the fingers the pinch does not use removes it entirely.

What remains is the recipe. It sets `rh_LFJ1/2/3` and `rh_MFJ*`/`rh_RFJ*` to
1.2/1.2/1.4 rad inside `initial_joint_targets` -- the **pregrasp** pose, before
the grasp begins -- which drives three unused fingers into the palm.

## This confirms plan section 4.3, option A

Option A proposes keeping the thumb/index active set and returning MF/RF/LF to a
canonical open posture instead of 1.2/1.2/1.4 rad. Measured directly:
`q_inactive_open` gives **zero self-contacts and zero penetration**.

One detail that matters for the fix: opening only the little finger is not
enough. `q_lf_zero` still leaves `rh_rfproximal` against the palm at 199.8 N.
All three inactive fingers have to open.

No mesh, contact material, safety budget or force threshold is touched by this,
which is what option A requires. Options B and C are not needed: B would need
the pair to be a structural adjacency artifact and it is not, and C would need
the collision proxy to be wrong and the neutral posture shows it is not.

## Not yet done

This is localization only. The recipe delta itself is P3.4.1-07, and its
regression blast radius across P2, P3.2 and P3.3 is P3.4.1-08 -- the Shadow
recipe is shared, so changing it re-opens claims those phases already closed.
