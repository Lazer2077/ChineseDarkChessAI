# Teacher-policy retention after neural self-play updates

The first function-level factorial contrast loses23W/0D/41L with the updated
policy and unchanged warm value, using matched two-forward inference. Increasing
training search effort from256to2048simulations also fails to improve the
warm-control screen. These results motivate checking policy supervision and
knowledge retention rather than assuming a critic-only bottleneck.

[Kickstarting Deep Reinforcement Learning](https://arxiv.org/pdf/1803.03835)
adds teacher-policy cross entropy to the RL objective on student-generated
trajectories, with an influence that can change during learning. It is useful
precedent for retaining a competent policy while optimizing reward, rather than
discarding teacher guidance immediately. The paper uses policy-gradient RL and
does not establish guarantees for our search-visit learner or Jieqi. Applying
its general distillation idea would be an attributed adaptation, not a new
algorithm by itself; a constant anchor would not reproduce its adaptive scheme.

Before changing training, `results/teacher-retention-diagnostic-plan.json`
freezes four already selected checkpoints: original warm, first256simulation
LeafBatch32RL,256simulation LeafBatch8RL, and2048simulation LeafBatch32RL.
Each is evaluated in FP32 on the same37,471positions from512original teacher
selection games. These are **reused checkpoint-selection data**, not newly
held-out test games. Warm checkpoint selection itself may favor the warm model
on this dataset. No model is fitted, selected or promoted by this diagnostic.

Report teacher-action CE/top1/top5, full2550-action KL from warm to each model,
and value error against original70%static+30%terminal targets, terminal-only
targets and static-only targets. Separate partial and fully public positions.
The full action distributions are not legal-masked policies; do not interpret
their entropy/KL as legal-action-only search divergence. Runs use one CPU thread
and no GPU, preserving all current actual-clock match GPU allocations.

Retention loss alone does not prove that imitating the old teacher is optimal,
or that teacher anchoring will improve play. A later intervention needs a fixed
control, compatible information inputs, new isolated match seeds and the native
teacher attribution control. Search-visit fitting and actor strength must still
be evaluated separately. No new training campaign is selected before the
diagnostic and remaining function-level contrasts finish.

## Completed descriptive retention matrix

| Frozen model | Teacher action CE | Top1 | Warm-to-model KL | Original mixed-value MSE |
| --- | ---: | ---: | ---: | ---: |
| Warm | 2.364835 | 35.1258% | 0 | 0.084464 |
| RL256,Leaf32 | 2.759078 | 25.1928% | 0.547188 | 0.179038 |
| RL256,Leaf8 | 2.438848 | 32.7427% | 0.142073 | 0.150245 |
| RL2048,Leaf32 | 2.443420 | 32.3397% | 0.152238 | 0.120398 |

All runs are complete and use the same37,471reused positions. The high-search
and small-batch recipes preserve more teacher-policy agreement than original
RL256/Leaf32, but their completed matches still show no established warm-control
gain. Agreement retention is therefore not sufficient evidence of strength.
Mixed-value targets reflect the original teacher's search/continuation, whereas
neural self-play terminal labels reflect its own policies; this column measures
retention of old supervision, not whether the new value is an unbiased or
universally incorrect estimate. No uncertainty or new generalization claim is
attached to these descriptive means. Evidence:
`results/20261027-teacher-retention-diagnostic.json`.

## Legal-anchor implementation preparation

`research/policy_anchor.py` implements KL from the frozen warm policy to the
student, each normalized only over observer-compatible legal actions. The
existing full-action search CE and terminal MSE remain separate objectives.
No old probability mass on illegal actions is preserved by the auxiliary loss.
Teacher logits are detached, masked actions have zero KL derivative and empty
legal sets reject. Two tests verify an analytic KL/gradient example and actual
private-capture observer-mask/visit-support checks.

Masks are reconstructed exclusively from actor InformationState plus one fixed
internal compatible-inventory sample, with full known rule history. This is
not a proof of inventory-invariant rules. All256original RL training games,
13,924positions, pass exact observer replay and stored visit-support checks;
no episode or action is filtered silently. There are450,663legal action entries,
414,635visited entries and267fully public positions. Packed masks, offsets,
episode ordering and source digests are archived in
`results/20261027-observer-policy-masks.json`.

`scripts/train_anchored_selfplay.py` makes explicit guarded source substitutions
to the pinned original trainer; no existing file or active recipe is edited.
It queries a fixed eval-mode warm teacher on the **same augmented student
inputs**, retaining the student's original train-mode normalization behavior.
Checkpoint selection remains the original unanchored selection CE+MSE. A positive
coefficient requires additional training forwards; equal optimizer updates do
not imply equal training FLOPs.

The CPU integration uses four existing TRAIN games and two different TRAIN
games for smoke selection, one epoch/three updates. Zero coefficient reproduces
the original selected NPZ arrays bitwise; a positive coefficient gives finite
nonzero KL and changed weights. This verifies only the small integration case,
not full-run reproduction, a chosen research coefficient or playing strength.
No formal new training campaign has started. Evidence:
`results/20261027-legal-anchor-integration.json`.

The zero-coefficient implementation has now also reproduced the **full**
original first RL update on GPU:256training games/13,924positions,32selection
games, four epochs and436updates. Every selected parameter/buffer array and the
entire selected NPZ digest exactly match original checkpoint
`168ae3d997d8c671acf91d36b73d23eeedfdfc48985fc2f6a91948526b4d7d98`.
This extends the implementation check beyond the small CPU integration. It
reuses the same training seed and does not warrant duplicate matches or a new
strength/robustness claim. Evidence:
`results/20261027-legal-anchor-zero-reproduction.json`. A nonzero-coefficient
research campaign still waits for the original function-level study to finish.
