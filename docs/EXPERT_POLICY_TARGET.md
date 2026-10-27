# Replacing learner search targets with observer-only expert actions

The completed annotation study supplies14,981compatible roots from280whole
learner episodes, with eight incompatible episodes quarantined. At all but67
accepted roots, the original learner search already visits the expert action.
The remaining question is whether stronger supervision changes what the neural
policy learns, rather than whether more actions merely receive visits.

This is an attributed expert-distillation experiment, motivated by learner-state
distribution shift in [DAgger](https://proceedings.mlr.press/v15/ross11a.html).
[Reanalyse/MuZero Unplugged](https://arxiv.org/abs/2104.06294) provides a separate
game-AI precedent for recomputing training targets on existing trajectories.
Our experiment uses an external native expert, not a learned dynamics model or
the MuZero algorithm. Neither paper guarantees improvement for this compressed
Jieqi observation encoder, approximate teacher or changing self-play opponents.

## One controlled change

Both arms start from the same original64-channel/15-block warm model, retain
the same observer encoder and train on exactly249TRAIN games/13,304positions.
The same31selection games/1,677positions supply descriptive loss curves. The
seven incompatible TRAIN games and one incompatible selection game are absent
from both arms; no partial prefix of a quarantined game is used.

The control retains the original normalized completed-search-visit policy
distribution. The intervention replaces it with the one-hot action selected
by the unchanged native expert at that exact observer-specific full history,
using the previously frozen8192-node query. Both use full2550-action policy CE
and the identical original learner terminal-return MSE. Expert cp/mate scores
are deliberately unused. Thus only the policy supervision changes; the weaker
learner's value-target distribution remains a limitation in both arms.

There is no additional warm-policy anchor, entropy coefficient, privileged
value feature, new rollout or teacher call during deployment. This is not an
on-policy policy-gradient method or a newly established DAgger convergence
result. The source terminal labels remain Yuan-rule labels even though expert
actions come from native search; accepted-root agreement does not establish
equivalence of all future counterfactual continuations.

## Frozen training and evaluation

Use one training seed20261027, four epochs, batch128, AdamWlr1e-5,
weight decay0.0001, gradient clipping1 and the original minibatch/mirror draws
and normalization updates. Each arm performs416updates. Export the fixed final
fourth epoch; profile-specific losses are logged but never select a different
checkpoint or duration. Loss magnitudes under soft visit targets and hard
expert targets are not comparable chess-strength estimates.

The unchanged-target CPU smoke exactly reproduces the original one-epoch,
three-update checkpoint `3aa9ffd099d5c41ef0f6df36d958cbf726b3c6904ea036024f7ee017076e1de0`.
The expert-target smoke completes the same budget with a different finite
checkpoint. These use four original TRAIN games for fitting and two additional
TRAIN games for smoke evaluation; they are implementation checks, not isolated
generalization or playing-strength evidence. Three tests verify exact split
roles and whole-episode exclusions, preservation of observations/terminal
targets, score-field independence and rejection of changed label digests.

Before full fitting, freeze three64-game development comparisons of the
intervention against the matched newly fitted control, original warm actor,
and unchanged native teacher: seeds9640001,9641001and9642001respectively.
All finish regardless of earlier results. Use actual10+0.1time control,
LeafBatch32for both neural actors, one CPU thread each, two concurrent games
and one shared dedicated inference GPU within a match campaign. Training uses
one GPU; the two arms run sequentially. There is no distributed-training or
equal FLOP/Hash/simulation claim.

Raw native-rule matches, complete observer histories, paired uncertainty and
native seeded-reveal/legality/terminal/clock audits remain mandatory. Preserve
technical failures and rounded-clock ambiguities as flags. Beating a degraded
control alone is insufficient; improvement must be measured against warm and
the unchanged teacher before promotion. This pilot cannot satisfy the final
independent multi-set70%strong-suite/report gate.

Protocol: `results/expert-policy-target-research-protocol.json`.
Implementation checks: `results/20261027-expert-policy-integration-{control,expert}.json`.
Frozen runs: `results/expert-policy-target-{control,expert}-training-plan.json`
and `results/expert-policy-target-training-pipeline-plan.json`.

## Completed training; matches in progress

Both profiles complete the fixed416updates and export epoch4. The matched
visit-target control has NPZ digest
`eff3901fa82f2b71ef1f17b9f1bfbeee00a4a06700e404f546b0bc0a51058c12`;
the observer-expert policy model has digest
`0fa5152fd4f4a59a93625579624daf642b9a9b78751f206723a0cdeec94404f8`.
No checkpoint is selected using the resulting match outcomes. Full curves,
initial/final losses, input hashes and exact split sizes are archived in
`results/20261027-expert-policy-target-{control,expert}-training.json`.

The three prespecified comparisons have started with those frozen models.
After fitting, a separate one-thread CPU diagnostic compares warm/control/expert
retention on the same repeatedly reused512teacher-selection games, including
observer-only entropy strata. It neither refits/selects actors nor establishes
independent generalization or legal-masked policy KL. It preserves the actual
clock-match GPU allocation and cannot establish playing strength.

Plans: `results/expert-policy-target-{evaluation,retention}-plan.json`.
