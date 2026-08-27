# P3.4-14 static-vs-dynamic ablation

## Verdict: `no_measured_difference`

| arm | candidates | yield | unsafe rejections | dominant failure |
| --- | --- | --- | --- | --- |
| static (target welded) | 6 | 0.0 | 0 | `insufficient_enclosure` |
| dynamic (target free) | 6 | 0.0 | 0 | `insufficient_enclosure` |

Yield delta: `0.0`. Config hash: `b108a5183d3b952a`.

## Reading this honestly

**The Phase 3.4 hypothesis is not confirmed by this run, and this run cannot
confirm it.** Both arms return zero. The scene is the micro pusher: one slide
actuator against a box on a table. It has no fingers, so it can never satisfy
the enclosure condition, and no arm can produce a positive regardless of whether
the target is free to move.

What the run does establish is that the ablation machinery is complete and
correct end to end: two arms differing only in the frozen-object assumption, six
candidates each fully accounted, a reason ledger with a denominator at every
stage, and a config hash over the budget, limits and seeds so nothing can be
retuned after the fact.

A real test of the hypothesis needs the release hands in scenes where a
static approach is blocked and extrinsic dexterity has something to offer.
That is `P3.4-16` work and it has not been done.

## What was deliberately not done

The obvious way to make this table look better is to relax the enclosure
requirement so the single pusher counts as a grasp. That would make the ablation
measure nothing. The thresholds in `RolloutLimits` were pinned before the run and
are hashed into the report.

## SHA-256

- `static-vs-dynamic.json`: `ec2a9cc918a3031d19634a14ded8f78dd02210e16b12b4a55f92132677c13980`
