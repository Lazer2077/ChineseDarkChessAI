# Self-play updates with the relative policy decoder

The wider decoder comparison is supervised teacher-replay pretraining. To enable
a later controlled RL iteration, new entrypoints preserve the original complete-
episode generator and search-visit/terminal-return optimizer while loading and
exporting either decoder safely. No main-study training, checkpoint selection
or fixed strength screen changes as part of this integration.

`generate_spatial_selfplay.py` uses the same attributed observer-information
PUCT, complete public/private rule histories, independent environment/search RNG,
Dirichlet/temperature recipe and exact completed visit counts as the original
generator. Its only model change is the tagged safe loader. The observed board,
inventory encoder and2550-action geometry are unchanged. The spawn-worker
initializer is a top-level callable; no graph inference or omniscient teacher
is added. Truncated episodes remain errors, never fabricated terminal targets.

`train_spatial_selfplay.py` preserves original AdamW/gradient clipping, horizontal
augmentation, visit-distribution cross entropy plus terminal-return MSE, and
selection-split checkpoint rule. It validates the planned decoder and preserves
decoder/mapping metadata in safe NPZ exports. This remains AlphaZero-style
search-policy/outcome fitting, not an asymmetric policy-gradient estimator.
The values describe the generating observer-limited policies under the source
rule profile; neither native teacher values nor hidden identities enter targets.

## Bounded integration before larger self-play

Three tiny8-channel/one-block profiles use six complete games at8simulations per
decision, LeafBatch32, one generation worker, the same six environment seeds,
four TRAIN/two selection games, one training seed and one CPU optimization epoch.
The first two profiles load the identical existing dense model through original
versus new entrypoints. Every trajectory array, outcome and final model NPZ must
match exactly. The relative profile must independently generate complete audited
episodes, update, preserve its policy-map metadata and safely reload finite
policy/value outputs. Matching seeds do not make its different trajectories a
paired strength test.

Every written/reloaded episode replays the observer records, actual reveals,
captor-private captures, legal visit support, completed root visits and terminal
perspective. Curves and complete episode archives are retained. These are
software/data-boundary controls, not new large-model RL, improving expected
returns or an opponent victory. In particular, source generation still uses
pinned Yuan tiantian-v5, whose termination/history profile differs from the
native-jqe benchmark. Passing its replay audit does not establish native-game
compatibility or an equilibrium guarantee.

Recipe: `results/spatial-selfplay-integration-pipeline-plan.json`.
The large128x15pretraining and its five frozen eager strength contrasts proceed
unchanged. Any subsequent large self-play recipe must bind the selected model,
fresh generation data, one training seed and independent strength-test layouts
before games; a successful integration alone does not justify promotion.

The integration is now complete:6/6/6games with1328/1328/708recorded positions
and9/9/12private dark captures for dense-original/dense-new/relative-new. The
two dense arms deliberately duplicate trajectories;3364records are not3364
independent or unique observations. Dense updates complete12steps each and
produce the identical NPZ digest
`1eafa8938e27aefc22817ce1dc024cf10c6f5dc9544f551a9bfbbbd469e58251`.
The relative update completes7steps, changes its initial model and safely
reloads as `relative50`, digest
`c1fdaabd675ef239e7ab01ad24f324d50f207ee620deb6515de030b7f3ef6ee7`.
All episode replays pass the listed source-rule/observer/visit/terminal audits.
Evidence, training curves and raw episode archive:
`results/20261027-spatial-selfplay-integration.json`.

## One separate full-size RL iteration completed: regression

The completed relative parent `e43ff2aa` now starts one prespecified research
iteration:288new self-play games at256completed simulations/LeafBatch32,
256TRAIN and32selection games, four epochs/batch128/lr1e-5and one training
seed20261031. Actual generation seeds begin9700001. Use the same original
search-visit and actor-relative source-rule terminal targets, normalization,
augmentation and optimizer arithmetic; eager FP32 inference stays unchanged.
This is new current-policy self-play RL, unlike the preceding teacher-replay
pretraining. It is not a privileged teacher or an unbiased policy-gradient
algorithm. Different model, generation/training seeds and resulting trajectories
prevent a clean architecture-only comparison with old small-model RL runs.

Every completed episode NPZ/JSON and the generation manifest are hashed into
the training plan before its first optimizer step. No truncations or invented
returns are accepted. Choose a checkpoint only from the declared32selection
games, then complete both64-game actual10+0.1screens at9720001versus the parent
and9721001versus the unchanged native teacher, LeafBatch32and2concurrent games.
Raw archives and independent native identity/legal/terminal/clock audits follow.
Neither screen selects another checkpoint or changes the other screen.

The relative branch and parent checkpoint are chosen from the specified
architecture/selection recipe; pretraining match scores have not been inspected
to choose this RL parent. All five original pretraining matches continue with
their already frozen parent models. This RL research actor is separate and has
no promotion or final-suite status. The known source/native rule-profile
difference remains explicit. Protocol and immutable runtime inputs:
`results/spatial-relative-rl-plan.json` and
`results/spatial-relative-rl-pipeline-plan.json`.

A separate queued archive waits for the original live controller to complete
both matches and all native audits. It then replays every source episode,
checks the complete optimizer/selection budget and exact played-model digest,
and retains raw episodes, curves and positive/negative/technical match flags.
It never restarts a missing or failed dependency. Recipe:
`results/spatial-relative-rl-archive-queue-plan.json`.

Generation and fitting have now completed, before either strength screen:
288games/16,009positions, including877private dark captures and446fully public
positions. The fixed whole-game split contains14,255TRAIN positions from256games
and1754selection positions from32games. All576episode NPZ/JSON artifacts plus
the completed generation manifest are bound in the training-plan digests.

All four epochs finish448updates (112per epoch). The fixed selection rule picks
epoch4, model
`1f6bf7acb49b23439104318c8bc31c196570f4bb4614164f39f75823ae379a2e`.
Selection search-policy CE changes5.735707→3.330671and source terminal-return
MSE0.944537→0.890993. These targets/data differ from teacher-replay pretraining;
their magnitudes are not directly comparable with its CE/mixed-value MSE. Lower
self-play selection loss does not prove policy improvement or native strength.
Both fixed native strength screens are complete with independent identity,
legality, terminal and actual-clock audits:

| Opponent | W/D/L | Score | Paired score bootstrap95% | Audited moves |
| --- | --- | ---: | --- | ---: |
| Frozen pretrained parent | 20/2/42 | 32.8125% | [21.875%,43.75%] | 3062 |
| Unchanged native teacher | 0/0/64 | 0% | [0%,0%] | 2879 |

There are zero technical games or clock ambiguities in these128games. This
particular update regresses against its parent despite lower selection losses.
The all-zero teacher bootstrap interval reflects the observed sample and is
not a claim that the true score is exactly zero. The branch is closed without
another unchanged iteration or promotion. Complete source episodes, both
training curves and match/audit digests are retained in
`results/20261027-spatial-relative-rl-training.json`.

A separate training-only public-native replay audit subsequently quantifies the
known rule-profile mismatch:19of256TRAIN games end earlier under native rules,
12have a different terminal return, and609source positions occur after the
native terminal. This diagnostic is not proof that the mismatch caused the
observed regression. A new matched prefix/return-label study tests that distinct
hypothesis without altering this completed experiment or its original labels;
see [native-terminal reanalysis](NATIVE_TERMINAL_REANALYSIS.md).

The independently specified [policy-target pruning study](POLICY_TARGET_PRUNING.md)
addresses exploratory visit targets on the same certified native prefixes.
Its32-root TRAIN teacher proxy supports a bounded matched learner comparison;
no resulting actor-strength gain is established yet.
