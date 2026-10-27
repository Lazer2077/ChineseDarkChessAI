# Legal warm-policy anchoring in search-guided RL

The original first search-guided RL update fits its new visit targets while
losing teacher-policy agreement and its warm-control match. Function-level
mixed policy/value actors both fail to improve the warm control. A full
zero-coefficient reproduction verifies that the optional anchoring code returns
the original selected arrays exactly. These motivate one bounded, constant-
coefficient auxiliary-policy experiment, without another data-generation run.

## Objective and information boundary

Use the original full2550-action search-visit CE and terminal-return MSE, adding

    L = CE(pi_search, pi_theta) + MSE(v_theta, z)
        + lambda KL(pi_warm,legal || pi_theta,legal),  lambda = 1.

Both policies in the KL are normalized over the same actor-compatible legal
set. The full-action CE still penalizes probability outside recorded search
support. The auxiliary loss does not retain old probability on illegal moves,
and its teacher gradients are stopped. It queries the frozen original warm
CNN, **not** the native NNUE/PVS teacher and not a clairvoyant policy. The warm
network itself was previously trained from the native teacher's lawful replay.

The frozen warm teacher receives exactly the student's augmented observation,
including horizontal mirrors. Teacher normalization is eval-mode; student
normalization retains original train-mode behavior. Full known rule history is
used when reconstructing legal masks, while both encoders omit unavailable
historical feature planes exactly as before. Masks use actor InformationState
and one fixed internal compatible-inventory sample, never the actual referee
pool; this is a disclosed observer-mask profile, not a universal rules-invariance
proof. Every stored visit and chosen action is checked against those masks.

The auxiliary teacher-policy idea is attributed to
[Kickstarting](https://arxiv.org/pdf/1803.03835). This constant one-update
search-visit adaptation is not its adaptive influence/PBT scheme, a new
policy-gradient theorem or an equilibrium algorithm.

## Frozen single-update recipe and completed training

Reuse the original256training games/13,924positions and32selection
games/1,758positions, original training seed and generating warm checkpoint,
four epochs, batch128, AdamWlr1e-5/weight decay0.0001, clipping1 and the
original mirror/order RNG. No new rollout, capacity change, privileged feature,
training-seed replication or coefficient sweep is introduced. The zero arm
already reproduced all436updates and checkpoint168ae3d9bitwise; identical
control weights do not need redundant matches.

The positive arm completed436updates and selected epoch4using the original
**unanchored** selection CE+MSE, as fixed before evaluation. Its selected digest:
`c09575bbee3d4d9f2c4dd6e6818b569f156b20973bd09b402e5f7b51eaf7cffb`.
Own-selection search CE4.0544217and terminal MSE0.7559217are not evidence of
strength; the original unanchored model has CE3.8773666and MSE0.7605823on the
same selection data. The objective balances visit fitting against a teacher-consistency penalty;
actual retention is measured separately.
Mean final-epoch training legal KL is0.272111. No outcome-based checkpoint
selection or promotion occurs.

Training teacher forwards add compute, but deployment remains one unchanged
CNN forward per leaf batch under the original PUCT. There is no extra teacher
query or added input at inference. Equal training updates do not mean equal
training FLOPs.

## Frozen independent-layout development matches

After selection and before the first game, evaluation pins the selected array,
metadata, initial/control artifacts, source files and research protocol. All
three64-game contrasts finish regardless of intermediate outcomes:

| Candidate | Opponent | Layout seed |
| --- | --- | --- |
| Anchored actor | Original unanchored RL1,168ae3d9 | 9610001 |
| Same anchored actor | Original warm,4ff3c675 | 9611001 |
| Same anchored actor | Unchanged native jqe+it014 | 9612001 |

All use actual10+0.1, LeafBatch32for both neural actors, one CPU thread each,
two concurrent games and one dedicated GPU shared by neural actors. The native
teacher is CPU-only. There is no equal-FLOP, simulation-count or supported Hash
memory claim. These are fresh development layouts, not final confirmation
sets. Native identity, legality, full private-history terminal and clock audits
plus raw archiving follow each completed contrast.

A separate one-CPU-thread descriptive retention run compares warm and the
already selected anchored model on the same reused512teacher-selection games.
It cannot select/retrain the model or replace fresh matches as evidence. No
benefit, teacher superiority or promotion is established while matches run.

Evidence: `results/legal-anchor-{research-protocol,training-plan,evaluation-plan,retention-plan}.json`;
training curve/checkpoint metadata: `results/20261027-legal-anchor-training.json`.

## Completed first contrast and retention diagnostic

Against original unanchored RL1, the first frozen64-game screen completed
43W/1D/20L:67.1875%wins,67.96875%score, paired bootstrap95% score interval
[55.46875%,79.6875%]. The conservative paired Hoeffding interval
[43.96068%,91.97682%] still includes50%. All3,243moves pass independent native
identity/legal/full-history terminal/clock replay, with no technical games or
ambiguous margins. This is a positive single development contrast against the
degraded RL1checkpoint, not evidence against the warm model or native teacher,
not final70%confirmation and not a promotion. Both other frozen screens continue.

On the reused teacher-selection data, anchored teacher-action CE is2.499138,
top1is31.3069%and warm-to-model full-action KL0.233183. The original unanchored
RL1has2.759078/25.1928%/0.547188, so the anchor retains more of the warm policy
on this diagnostic, while still trailing the original warm35.1258%top1.
Old mixed-value supervision MSE is0.162651versus0.179038for unanchored and
0.084464for warm. These reused-selection means do not establish independent
generalization or explain the match difference causally by themselves.

Evidence: `results/20261027-legal-anchor-vs-unanchored{,-audit}.json` and
`results/20261027-legal-anchor-retention.json`. A separate
[uncertainty diagnostic](OBSERVATION_UNCERTAINTY.md) prepares observable stage
features only; no change to this running recipe is made.
