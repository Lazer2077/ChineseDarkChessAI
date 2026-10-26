# Research progress — 2026-10-26

Active goal is in GOAL.md. No baseline is marked defeated.

## Pilot results
The integer-feature training failure was fixed with explicit float32 tensors;
a synthetic all-integer row now trains successfully. Both 20-epoch stages then
finished on GPU 2, with 997 and 613 search-labelled rows respectively.

Root-sampling trained prototype against pinned mist-jieqi JS: 1 win, 1 draw,
6 losses over 8 games (4 paired layouts), score 18.75%. No truncation. Against
PikaJieQi: 0 wins / 0 draws / 2 losses; against jqe with its published it014
network: 0 / 0 / 2. These are small **unequal-budget integration pilots**, not
formal strength comparisons. Failures are evidence to improve search/evaluation.

The pinned jqe diagnostic agreed with our inherited legal move generator on
2,228 positions from 20 seeded random games. No mismatch. This only checks legal
geometry: adjudication is deliberately NOT considered compatible. jqe has
perpetual check/chase rules and an 80-ply no-capture clock; our current pilot has
third-repetition responsibility and a 120-ply no-progress clock.

## Next experiment
A bounded native-teacher pilot generates 512 training and 128 validation games,
2,000 search nodes per move, two CPU workers; then trains an NNUE for 10 epochs
on GPU 2 using the external MIT jqe trainer. Training/validation seeds are
separate. This is a feasibility experiment for a stronger existing architecture,
not a claim that jqe is our own contribution. Native data-generation/adjudication
limitations must be audited before using those labels in formal research.

The PikaJieQi build emitted an upstream array-bound/undefined-behaviour warning
in Zobrist initialization (position.cpp). It ran the smoke matches; this does not
resolve the warning. Keep the pinned source unchanged and audit before formal runs.

Local evidence: runs/first-experiment, runs/pika-adapter-smoke,
runs/jqe-adapter-smoke, runs/rule-oracle, and runs/native-teacher-pilot.

## Native algorithm and coverage update
512 native teacher games yielded 34,964 records (23,676 training rows after
filtering); 128 held-out games yielded 5,882 retained validation rows.
The trained pilot scored 7-0-25 against its teacher at equal 100 ms/move, so it
was not promoted. New probe-variance search scored 14-2-16 against unchanged jqe
with the same published net; also not promoted. See ADAPTIVE_SEARCH.md for
methods, build reproducibility, provenance and completed opponent screens.

## Continual learning series started
A two-iteration functional smoke verified replay, GPU value learning, technical
failure rejection and idempotent recovery. Three forced truncated games wrote
zero records. The three-seed expanded series is live under
runs/continual-series-01 (49,152 training games planned; not completed).
The first seed's 512 validation games finished: 37,471 raw records, zero rejected
games. Its first 8,192-game training generation is now running on four CPU actors;
GPU 2 will perform the subsequent batch learning. Separate 1.64M/13.45M capacity
initializations passed function-preservation and native benchmark checks but
have not yet been trained. No opponent or new candidate is promoted.
