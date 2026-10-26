# Controlled increase in self-play search effort

The first two256-simulation neural RL updates regressed against their preceding
actors. Frozen normalization statistics also regressed against the original warm
model. Those results leave visit-target/search quality as an unresolved bottleneck;
they do not establish it as the cause.

`results/neural-high-search-plan.json` predeclares one update from the same warm
checkpoint with2048 completed simulations per move. It retains the original
first iteration's256 training/32 selection episodes, environment and training
seeds, LeafBatch32, original normalization behavior, four epochs, batch128 and
learning rate1e-5. More search changes trajectories and therefore positions,
terminal labels and possibly optimizer step counts; matching environment seeds
does not pair identical trajectories. This isolates the search-effort recipe
intervention without claiming equal training compute or identical data.

One otherwise free GPU runs four inference workers, then training and two fixed
fresh64-game development matches against the warm checkpoint and native teacher.
Both match profiles retain actual10+0.1 clocks and LeafBatch32. This tests whether
more search during learning produces a better fixed-clock actor. It does not
give the candidate more inference time, add the experimental inventory filter,
or use the new smaller-batch search variant. The original source and all models
are pinned before/after stages. Native legal/private-history/terminal/clock
audits are required after completed matches before interpreting their results.

The run is active; there is no strength result or promotion. The first launch
failed during import before creating any run state because a native-library
environment variable was misspelled. The corrected launch uses the pinned
native rules/legality libraries. No live job was restarted and no data were
overwritten.
