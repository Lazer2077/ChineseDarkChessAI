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
