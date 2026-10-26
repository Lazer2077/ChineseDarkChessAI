# Confirmed version-specific results

Two pinned opponent versions now satisfy their fixed two-set confirmation
criteria. This is a benchmark result for the stated referee/clock and adapter
profiles, not a claim about all versions, rules, time controls, or all open-source
Jieqi engines. The candidate still has no confirmed improvement over its own
upstream teacher. The overall research goal remains incomplete.

The candidate uses unchanged attributed jqe search and the unpromoted expanded
small NNUE, SHA256
`9cd00a5f4fb7ccceb5c1374660a623acb60cd18eafa79f4c6aa9848b0e278833`.
Architecture, trainer and initial network are upstream contributions. Our
training recipe and negative controls remain part of the evidence.

| Pinned opponent | Set A W/D/L (score) | Set B W/D/L (score) |
| --- | --- | --- |
| AB-JChess v0.2d, `fe35d87c` | 162/7/31 (82.75%) | 150/2/48 (75.5%) |
| universe-st/Jieqi, `e00b667c` | 200/0/0 (100%) | 198/2/0 (99.5%) |

Each set has 100 distinct hidden layouts, played with colors swapped: 200 games.
The two sets use disjoint deal seeds and are assessed separately. Both players
receive 5+0.05 actual game clocks, and all position-to-bestmove wall time is
charged. The external referee is pinned jqe native: 80 no-capture plies and
public repetition/check/chase adjudication. Each player receives only its own
complete reveal/capture history; no hidden deal is passed to either engine.

The pinned upstream checkouts remain unchanged. The AB compiled copy has the
documented GCC11 declaration-only constexpr-to-inline compatibility repair;
function bodies, search and evaluation are unchanged. AB uses its official
released network. Universe uses its original hard search through the
documented observer/timing adapter, with original data structures; it ignores
Hash/Threads settings, so equal hash memory is not claimed. Internal search
repetition heuristics differ from the external referee, as documented in each
profile. This limitation is part of the result, not evidence of universal rule
equivalence.

Each set must have no technical failure or infrastructure void, and both the
paired bootstrap and conservative paired Hoeffding 95% score lower bounds must
exceed 50%. For AB these conservative lower bounds are 69.17% and 61.92%; for
Universe they are 86.42% and 85.92%. The perfect-score bootstrap interval in
Universe set A is degenerate [1,1]; the conservative bound prevents treating it
as certainty of perfect future performance. These are per-set/per-opponent
intervals, not a family-wise error guarantee over the entire evolving inventory.

Fresh seed-based replay checked all 27,768 AB learner moves and 12,384 Universe
moves, their public reveals, captures, legal moves, terminal outcomes and actual
clock charges. No rounding ambiguity was found. Source, runtime, opponent,
candidate and network hashes were checked after both sets completed.

Acceptance evidence independently recomputes WDL, paired intervals, color pairs,
disjoint layout seeds and consistency with replay audits:

- [AB acceptance](../results/20261026-abj-acceptance.json)
- [Universe acceptance](../results/20261026-universe-acceptance.json)

Compressed raw games, fixed plans, individual results and audits are linked from
those artifacts. `scripts/accept_native_opponent.py` checks one pinned profile;
it cannot complete the global goal or promote the learner over its teacher.

## What the teacher controls change

All four frozen AB arms completed, including the unchanged teacher on the same
layouts. Teacher A scored 149/4/47 (75.5%); teacher B scored 157/4/39 (79.5%).
The trained candidate's difference is +7.25 percentage points on A and -4.00
on B. Across both sets the difference is +1.625 points, with exploratory
stratified common-layout bootstrap 95% [-3.125, +6.25] points. Its conservative
range-two Hoeffding interval also includes zero.

Thus superiority over AB does not establish a learned improvement over jqe.
The backbone already supplies substantial playing strength. This comparison
subtracts each control's score against AB, after averaging each color-swapped
layout pair. It is not a direct learner-versus-teacher match, and sequential arm
order/shared-host scheduling were not randomized. Model selection is not
corrected by these exploratory intervals. The candidate's direct teacher gate
also failed. No global actor promotion follows this milestone.

Contrast recipe and evidence:
`scripts/compare_native_controls.py`,
[common-layout contrast](../results/20261026-abj-teacher-contrast.json).

## Remaining coverage and negative evidence

The classical Pika original's first set passed, but its second set includes an
opponent time forfeit; the two-set criterion is not met. The new Pika/ABJ-NNUE
hybrid has engine-exit voids and cannot pass a clean-run gate even with a high
score. Mist's first actual-clock set also contains opponent time forfeits.
These are not marked defeated. Their fixed campaigns retain every failure and
continue without retuning. Exact failure-prefix diagnostic replays do not replace
failed match attempts or certify their cause.

The wider inventory includes unresolved licenses, unavailable weights,
incompatible variants and unverified repository leads. See
[the inventory](opponents.json); the two accepted versions do not exhaust it.
