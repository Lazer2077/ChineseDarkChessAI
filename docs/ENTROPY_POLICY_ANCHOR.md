# Observable-entropy allocation of a fixed mean teacher constraint

The constant legal-policy anchor beats the degraded first RL model in one
development contrast, but does not beat the original warm model and loses all
64games against the native teacher. After that full campaign closes, test the
previously proposed observable stage weighting without increasing its mean
coefficient or changing the training trajectories.

## Controlled intervention

For each original TRAIN record compute normalized live-type assignment entropy
h_i from the actor's public board and known inventories, under the existing
exchangeable count prior. Set

    weight_i = h_i / mean_TRAIN(h),
    L_anchor = mean_batch(weight_i * KL(warm_legal || student_legal)).

Mean coefficient is1over all13,924TRAIN positions, matching the constant-unit
baseline. The coefficient is not normalized again within a minibatch and does
not receive gradients. Its recorded float32range is[0,2.70657778]. Thus the
same average teacher constraint is allocated more heavily to uncertain states
and removed at type certainty, including some still-face-down positions.
Neither public-board canonicalization nor move geometry changes.

This is a falsifiable weighting heuristic. Live-type entropy is not uncertainty
about every opponent belief/history, a confidence estimate for the neural value
function, or a guarantee of safer policy targets. The completed-Q diagnostic
already illustrates that approximate action values can be misleading in a
fully public root. Equal mean coefficients do not imply equal gradients,
realized penalty or effective learning strength. No performance claim follows
from their normalization.

## Full control and selected actor

Before fitting the new profile, the extended trainer repeats the complete
constant-unit update:256training games/13,924positions,32selection games,
four epochs and436updates. Its selected NPZ digest is exactly
`c09575bbee3d4d9f2c4dd6e6818b569f156b20973bd09b402e5f7b51eaf7cffb`,
the established constant anchor. Failure of this check stops the pipeline.
This is same-seed implementation validation, not another training-seed success
or new chess result.

The entropy profile reuses the same initial warm network, seed, augmentation,
optimizer, legal masks, replay/selection split, clipping and number of updates.
Only record-dependent teacher coefficients change. Teacher queries remain on
the same mirrored student inputs. It completes436updates and selects epoch4by
the original unweighted search CE+terminal MSE, independently of any matches.

Selected digest:
`567043dc673a055bd7ea064435dbdc3571d894aba4506ca8aa763c269810064e`.
Own selection CE4.1186675and terminal MSE0.7595189do not establish improvement
over the constant model or original warm checkpoint. No new training seed,
rollout corpus, coefficient sweep, model size or runtime teacher query is added.
Deployment remains the same one-network PUCT actor/LeafBatch32.

## Frozen development evaluation

After selection and before the first game, freeze all selected artifacts,
source hashes and three fresh64-game paired contrasts:

| Opponent | Layout seed |
| --- | --- |
| Constant-unit anchor,c09575bb | 9630001 |
| Original warm,4ff3c675 | 9631001 |
| Unchanged native jqe+it014 | 9632001 |

Actual10+0.1, one CPU thread each, two concurrent games and a dedicated shared
GPU for neural actors. No equal simulation/FLOP/Hash memory claim. All three
contrasts finish regardless of intermediate outcomes, with native identity,
legal/private-history terminal and actual-clock audits plus raw archives.
They are development screens, not final70%confirmation or promotion.

A separate one-CPU-thread retention analysis compares already selected warm,
constant and entropy actors on the same reused512teacher-selection games,
including observer-only entropy strata. It neither selects/refits weights nor
establishes independent generalization; reported KL is over full2550-action
distributions, not the legal-only training objective.

Evidence: `results/entropy-anchor-{research-protocol,training-pipeline-plan,evaluation-plan,retention-plan}.json`;
full control: `results/20261027-entropy-anchor-constant-reproduction.json`;
training curve and weighting metadata: `results/20261027-entropy-anchor-training.json`.
