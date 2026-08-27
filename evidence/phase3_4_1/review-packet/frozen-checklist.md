# Frozen review checklist -- Phase 3.4 / 3.4.1

Frozen from `ROADMAP-P3.4-001` section 16 and `ROADMAP-P3.4.1-001` section 5.2.
Every item is written so that "no" is a possible answer.

1. [ ] The 2.0x speed gate was not changed, and 4.444x is not presented as a full pass while 29 worlds went NaN.
2. [ ] VRAM is measured outside the PyTorch allocator; the earlier 0.0 GiB figure is marked invalid rather than deleted.
3. [ ] nonfinite, overflow and OOM are reported as three separate quantities.
4. [ ] GPU and CPU run the same workload, and every finalist is replayed on the CPU oracle.
5. [ ] The reject denominator is reported, not only the accepted count.
6. [ ] No safety threshold, contact stiffness, actuator gain or force budget was lowered anywhere in Phase 3.4 or 3.4.1.
7. [ ] The Shadow classification is supported by measurement, and no legitimate collision was disabled to reach it.
8. [ ] Impacted P2, P3.2 and P3.3 claims are either untouched or replayed; the Shadow recipe is shared.
9. [ ] Evidence hashes match the exact commit and the worktree is clean.
10. [ ] Every failure is included in the packet, not only the runs that passed.
11. [ ] Findings at severity S0 to S3 are resolved, or the verdict is not a pass.

## Author disclosures

The author declares, and the reviewer should verify rather than accept:

- safety thresholds changed: **none**
- objective weights changed after seeing results: **none**
- gates relaxed to pass: **none**

Budget change, disclosed in full: one: contact impulse is judged over a rolling window instead of accumulated over the whole rollout. Applied identically to all three hands. Rationale: a cumulative impulse limit rejects every sustained hold regardless of how gentle it is, so it measured grasp duration rather than safety. LEAP and Allegro pass under the same budget Shadow fails.

Operating point change, disclosed in full: yes: from 64 worlds to 1024. 64 used ~0 GiB of a 14 GiB budget, leaving the GPU idle. 1024 was declared once before running and both counts are reported. The batch was not looped upward until a number passed.

## Known unresolved at the time of packaging

- 29-34 of 1024 GPU worlds go non-finite from identical inputs; all 990 survivors also differ from each other, so the divergence is not confined to the rejected worlds.
- shadow_hand has no dynamic positive; option A fixes the safety violation but the sweep parameterised tendon-coupled joints as independent.
- Data.overflow has never been read on a GPU run.

## Verdict

Left empty. The author wrote every artifact in this packet and cannot sign it.
Section 5.2 requires a reviewer in a separate context who does not edit the
artifacts and is not handed a desired conclusion.

```
verdict:          PASS | FAIL | BLOCKED
reviewer_type:    external | internal_independent
reviewer:
date:
findings S0-S3:
```

Any change to the artifacts after a verdict invalidates it.
