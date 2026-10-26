# Phase-conditioned validation diagnostics

The first full-public feature ablations did not establish stronger play. Before
adding further capacity or target mixtures, we measured where the models change
their value predictions. This is diagnostic work on the existing checkpoint
selection set, not a new held-out strength test.

`scripts/phase_value_diagnostics.py` verifies finalized trajectory hashes and
game boundaries, applies the native trainer's filtering, and evaluates raw or
full-public-canonical inputs according to each model's actual inference policy.
It retains the same 70% search / 30% outcome validation target in every arm.
No private identity unavailable to the acting player is supplied as input.

Of 512 games and 25,031 retained positions, only 88 games contribute the 3,061
retained fully public positions. There are 287 games with 1–4 hidden pieces,
491 with 5–14, and 512 with 15–30. These groups overlap across a game's lifetime.
Game counts refer to retained positions: a game without retained full-public
positions need not have terminated before every dark piece disappeared.

| Model / inference | All-position mixed MSE | Full-public mixed MSE | Full-public outcome Brier |
| --- | ---: | ---: | ---: |
| Original teacher / raw | 0.0141831 | 0.0075323 | 0.0436315 |
| Expanded small / raw | 0.0141022 | 0.0074310 | 0.0430049 |
| Canonical retraining / canonical | 0.0142397 | 0.0083387 | 0.0422589 |
| Two-ply temporal / raw | 0.0141188 | 0.0074658 | 0.0434259 |

Canonical retraining improves the full-public outcome Brier point estimate
while worsening its mixed-target position MSE. This illustrates that outcome
prediction, approximate search imitation, and playing strength are different
objectives; none of these point estimates demonstrates a strength gain.

Intervals in the raw report compare each model with the teacher, first averaging
error differences within each game and phase, then bootstrapping complete games
with equal game weight. They consequently estimate a different quantity from
position-weighted MSE in the table. Games with no retained position in a phase
do not enter that phase's conditional estimate. The reference model has exact
zero difference by construction. Checkpoint selection, several models and
several phases make these exploratory intervals unsuitable as confirmatory
significance claims. No multiplicity correction or new independent test is
claimed. Evidence: results/20261026-phase-value-diagnostics.json.

## Consequences for the next experiment

A targeted full-public continuation benchmark should preserve public repetition,
check/chase history and counters. Resetting these at a sampled board would change
the task. Select at most one transition per source game before looking at model
outcomes, pair engine assignments on the same history, and report source-game
counts as well as position counts. Freeze layout/source-game splits and budgets;
keep development diagnostics separate from later strength confirmation.

[TDLeaf](https://arxiv.org/abs/cs/9901001) motivates learning with search-derived
values, but our tested two-ply mixture is not its leaf-gradient algorithm.
[ReBeL](https://arxiv.org/abs/2007.13544) and
[Student of Games](https://arxiv.org/abs/2112.03178) motivate reasoning over
information states when combining search and learning. They do not justify
treating one player's zero inventory entropy as common perfect information.
For Jieqi, switching to a deterministic continuation must be justified by
future-relevant public certainty while retaining the rule state. A remaining
unknown historical capture can still affect a fixed opponent's policy even
when it no longer changes the legal continuation game graph. These are design
constraints for future experiments, not implementations of those algorithms or
inherited convergence guarantees.
