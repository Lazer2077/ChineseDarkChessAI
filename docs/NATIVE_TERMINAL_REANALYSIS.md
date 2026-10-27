# Native-terminal reanalysis of source-rule self-play

The relative-decoder self-play update regressed against its frozen pretrained
parent. A training-only audit now quantifies a concrete objective mismatch:
source episodes can continue after the native benchmark has already ended,
and their later source winner can disagree with the native result. This is a
rule-alignment finding, not evidence of a stronger actor or a demonstrated
cause of the regression.

## Public-history diagnosis

Replay each complete source trajectory from the public starting board, retaining
every public move and actual mover reveal. The pinned native referee uses its
80-ply no-capture and repetition/check/chase adjudication. No actual hidden
assignment, game-seed completion, future reveals or reset-to-current-FEN shortcut
is supplied. The original source episode loader still verifies observer records,
captor-private captures, source legal visits and source terminal perspectives.

The256TRAIN-game audit covers14,255source positions. There are237fully equivalent
complete traces and19earlier native terminals, leaving609source positions after
native termination. Twelve games have different source/native terminal returns.
All13,646replayed native-prefix moves are legal; visited-action supports and
public board/turn/counter comparisons show no discrepancy on these roots.
This checks actual public histories, not all counterfactual legal masks or a
belief posterior theorem.

For example, source game1ends at ply101with a red source win. Native public
history adjudication ends at ply69with a red loss, with a perpetual-check
diagnostic. Fitting its later source return therefore contradicts the benchmark
objective at earlier roots. The8-game preflight is a subset of the256-game audit,
not an independent confirmation.

Evidence: `results/20261027-spatial-relative-rl-native-rule-{preflight,audit}.json`.

## Frozen transform and matched controls

Before new fits, freeze one transform for all288original games, retaining the
same256TRAIN/32selection identities. Admit only a positive-length prefix that
actually reaches a native terminal, has lawful chosen/visited actions and agrees
on the public board, turn and counters. Quarantine an unsupported whole game;
never manufacture a return at a nonterminal cutoff. Check each admitted terminal
again with the separate one-shot full-history native replay API. Both APIs use
the same pinned referee; this is not an independent rules implementation.

All288games are admitted, with15,331prefix positions. Twenty-one end earlier;
13have different terminal returns, and678post-terminal source positions are
removed. The21earlier adjudications include20single-color perpetual-check losses
and one chase-required diagnostic, resolved by the pinned native game referee.
Original source files remain immutable and retain their original labels.

| Profile | Observation/visit data | Return target | Updates |
| --- | --- | --- | ---: |
| Source full control | Complete source episodes | Later source terminal | 448 |
| Source prefix control | Actual native-terminal prefixes | Later source terminal, intentional ablation | 428 |
| Native prefix | Identical native-terminal prefixes | Actual native terminal | 428 |

The two prefix profiles have bitwise-identical observations and normalized
actual visit counts, game roles, order/augmentation streams and optimizer budget.
Only their return labels differ. The full-source versus prefix comparison also
changes data and update count, so it cannot isolate the return-label effect.
No prefix file is falsely serialized as an original source-terminal episode.

The data preflight independently projects returns using red-first turn parity.
The two prefix profiles have13,646TRAIN and1685selection positions;805TRAIN
and56selection returns differ. Search policies still come from the original
source-model PUCT. Native reanalysis fixes realized episode termination and
returns, not counterfactual search dynamics or posterior estimation.

## Fits complete; independent strength screens running

All three fits use the same pretrained relative128x15parent, one seed20261031,
four epochs, batch128, AdamWlr1e-5, the original FP32 arithmetic and horizontal
augmentation. Each uses the fixed final epoch4; selection losses are descriptive
and do not choose another checkpoint. The full-source control must exactly
reproduce the completed original RL model before either prefix fit.

| Profile | Selected NPZ SHA256 |
| --- | --- |
| Source full control | `1f6bf7acb49b23439104318c8bc31c196570f4bb4614164f39f75823ae379a2e` |
| Source prefix control | `f3ebb9132dc748840b66caa5cd76a46a440baa02a748e86ad168084e0e1f5f9c` |
| Native prefix | `318325e35e1a49b7418ed4c0514ee5fae9eaafd00596a72659d7811bc8cecd80` |

The full-source exact-reproduction gate passes. Both prefix fits finish428updates
and safely reload the relative50model with5,453,555parameters. Lower losses on
different return targets cannot establish better playing strength.

Freeze all model hashes before four64-game screens: native prefix versus source
prefix at9730001, source full at9731001, pretrained parent at9732001and unchanged
native teacher at9733001. Use actual10+0.1, LeafBatch32for neural actors, one CPU
thread each and two concurrent games on a shared inference device. Complete all
four contrasts regardless of results, with independent native identity/legal/
terminal/clock audits and raw archives. No match result can select another model,
alter the target transform or remove an unfavorable contrast.

This is a bounded reused-rollout diagnostic. It neither establishes native-model
search nor solves two-player partial observation. Any gain against a degraded
control alone is insufficient for promotion; the frozen parent and unchanged
teacher remain required attribution comparisons. No final-suite or report claim
follows from data correctness or successful fitting.

Evidence: `results/native-terminal-target-research-protocol.json`,
`results/20261027-native-terminal-annotations.json`,
`results/20261027-native-terminal-target-data-preflight.json`,
`results/20261027-native-terminal-{source-full-control,source-prefix-control,native-prefix}-training.json`
and `results/native-terminal-target-evaluation-plan.json`.

The first fixed contrast is complete: native versus source-return prefix control
scores32wins/3draws/29losses,52.34375%score with paired bootstrap95%
[41.40625%,63.28125%]. All64games/3339moves pass native identity/legal/terminal
and actual-clock audits, with zero technical games or rounding ambiguities.
This label-only screen is inconclusive. Native prefix versus the full-source
control also finishes at30wins/0draws/34losses,46.875%score with paired95%
[35.9375%,57.8125%], again technically clean. That second contrast changes
data/update count as declared and is not label-only. The remaining parent and
teacher screens continue unchanged. Neither finished comparison establishes
that correcting native returns alone
recovers the pretrained parent's strength. Evidence:
`results/20261027-native-terminal-native-prefix-vs-{source-prefix-control,source-full-control}{,-audit}.json`.
