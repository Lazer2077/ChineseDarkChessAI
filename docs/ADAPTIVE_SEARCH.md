# Probe-variance allocation v1 — development experiment

The initial root-sampling/linear-policy pilot lost heavily to the reference
engines. The next research path reuses the fixed MIT jqe native architecture
while developing explicit chance-node search changes. The unchanged jqe engine,
its published net and its results are not claimed as our original algorithm.

## Hypothesis and change
For each hidden-identity outcome, the backbone already computes a shallow value
in expected-score space. Its default then fully searches the most numerous
identity. CDCA ranks identities by `p_i * (v_i - E[v])^2` and searches the largest
probe-variance contributor when the weighted standard deviation exceeds 0.05.
A rare dangerous outcome can therefore receive deep search before a common
benign outcome. The full mean remains weighted by the visible urn counts.
The optional CDCAExtra flag adds one additional deep outcome (off initially).
This is a compute-allocation heuristic; it is not a posterior confidence bound,
complete POMCP implementation or equilibrium guarantee. Poor shallow probes may
misdirect the allocation. Only current observer-visible urns and hypothetical
reveals enter it; the engine receives no true hidden deal.

- Source patch: scripts/build_native.py, guarded by exact source anchors and SHA.
- Standalone decision logic: native/chance_allocation.h.
- Native backbone: devinokeefe/jieqi 331a6e612a038c4a04cf980f7b9482910fd50078 (MIT).
- CDCAAdaptive=0 is the unmodified search control; default experimental value is 1.
- CDCAThreshold=50 means a 0.05 expected-score standard deviation.
- CDCAExtra=0 initially; no additional full-depth outcome budget.

## Validation before games
Allocation tests cover rare influential outcomes, permutation consistency,
constant values, thresholds, single-outcome pools and invalid inputs.
The six depth-5 native benchmark rows (move, score, nodes) are identical to the
unmodified engine with CDCAAdaptive=0: 12,138 total nodes. Enabling v1 changes
search to 15,423 nodes, so runtime overhead must be evaluated by equal clocks.
Independent native referee geometry/repetition/chase/session tests all pass.

## Predeclared development tests
1. Adaptive v1 versus unchanged native backbone, published it014 net on both:
   32 games, 100 ms/move, 1 thread, 64 MB hash each, 2 concurrent games,
   seed 6100001, complete paired layouts and actor-specific move histories.
2. Newly trained native pilot net versus published it014, unchanged backbone
   both sides: same budgets, 32 games, independent seed 6200001.

These are separate experiments: search allocation and teacher distillation.
They do not establish advantages against the entire opponent set.
Two launch attempts failed before any game because Python 3.10 lacks StrEnum;
the preserved failed directories contain no results. The live run uses Python
3.11 with unchanged source and the same preregistered development seed.

If a change looks promising, freeze it and test >=100 new paired layouts with
another seed, report paired confidence and technical failures, and repeat on
another layout set before promotion. No tuning on confirmation layouts.

## Reproduce
```sh
python3 scripts/fetch_sources.py devinokeefe-jieqi
python3 scripts/build_native.py
cmake -S external/devinokeefe-jieqi/engine -B external/devinokeefe-jieqi/engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/devinokeefe-jieqi/engine/build -j2 --target jqe
cmake -S external/devinokeefe-jieqi -B external/devinokeefe-jieqi/build/release -DCMAKE_BUILD_TYPE=Release
cmake --build external/devinokeefe-jieqi/build/release -j2
ctest --test-dir external/devinokeefe-jieqi/build/release --output-on-failure
python3 scripts/check_native_ablation.py
python3.11 scripts/native_match.py --candidate=build/adaptive/bin/jqe --baseline=external/devinokeefe-jieqi/engine/build/jqe --net=external/devinokeefe-jieqi/engine/nets/it014.jqn --games=32 --movetime=100 --seed=6100001 --out=runs/your-new-run
```
Build the unchanged baseline separately with CMake from its `engine` directory
before ablation or a match. Never replace the baseline binary with our patched one.

## Completed development evidence
Adaptive v1: 14 wins, 2 draws, 16 losses (46.875% score), all 16 pairs complete,
no void/illegal/time results. No promotion. Newly trained native pilot:
7 wins, 0 draws, 25 losses (21.875%), no technical failures. No promotion.
These negative results are preserved; neither a validation-loss decrease nor
inheriting a strong backbone is evidence of improving on the teacher.

The standalone header and experiment scripts are ours. Native geometry, neural
architecture, baseline weights, trainer and referee are reused with attribution.

## Statistical guard
Paired bootstrap intervals can collapse to [1,1] when every observed pair wins.
New reports therefore also include a distribution-free Hoeffding interval over
independent bounded pair scores: mean ± sqrt(log(40)/(2*n_pairs)). A superiority
screen requires both lower bounds above 0.5, complete pairs and no technical
failures. A single report never grants overall promotion: independent confirmation
sets remain required. Under the subsequent user amendment in `GOAL.md`, a
single training seed is sufficient; multiple independent test/layout seeds,
using the same frozen checkpoint, replace the former training-seed prerequisite.

## Opponent coverage screen
The adaptive-v1 derived engine uses the published jqe net in this screen (its
hash is explicit). It is not the weak newly trained pilot. At 200 ms/move,
16 games against universe-st hard completed 16 wins; against original Pikafish
jieqi_old, 10 wins and 6 losses. These are eight pairs per opponent, insufficient
for superiority claims. The full-history musesfish adapter initially rejected
classical x-padding; all 32 failed attempts were recorded as void, not wins.
A serialization-only strict profile repaired this: a fresh two-game smoke
completed with two wins and no technical errors. Expanded screens are running.
No opponent is yet marked defeated.

## Additional completed screens
At 200 ms/move, one thread, same pinned native referee and full private histories:
- musesfish strict adapter: 16 wins / 0 draws / 0 losses, 8 pairs.
- universe-st hard: 16 / 0 / 0, 8 pairs.
- Pikafish jieqi_old: 10 / 0 / 6, 8 pairs.
- pikajieqi-enhanced: 13 / 0 / 3, 8 pairs.
- Mistboard PikaJieQi: 15 / 0 / 1, 8 pairs.
- AB-JChess v0.2d plus official V11 net: 1 / 0 / 3, 2 pairs.
No void/illegal/timeout results in these completed runs. The failed musesfish
protocol run is retained separately and is not mixed into the successful run.
The AB release builds with a documented declaration-only constexpr repair for
GCC11, no function-body/search/weight changes; release archive and net hashes
are recorded. Network load may add first-move overhead; these movetime screens
are not strict wall-clock tournament certification. Equal thread counts and go
budgets alone do not establish identical actual elapsed time.

Compact completed evidence: results/20261026-native-development.json.
None reaches the required independent confirmation volume. No superiority claim.

## Warm-start follow-up
The from-scratch net lost heavily on limited data. The follow-up imports the
published JQN1 network into a trainable state with byte-identical export, then
fine-tunes on the same disjoint training/validation sets with lr=1e-5, lambda=0.9,
K=430, batch=1024, 10 epochs, seed=20261026. Held-out loss moves from 0.004957 to
about 0.004909; this is not a strength claim. Unchanged native backbone with the
warm candidate faces the published teacher net for 32 new development games,
100 ms/move, seed 7000001. No adaptive search change in that comparison.

## Warm candidate independent confirmation plan
The warm candidate finished its 32-game development run 18 wins / 1 draw /
13 losses (57.8125%), no technical failures. It is not promoted. The following
confirmation was frozen before games; see results/warm-confirmation-plan.json
for candidate and baseline hashes.

Two new independent sets of 200 games (100 layout pairs each), seed 7200001 then
7300001, unchanged native search on both sides, one thread and 64 MB hash each,
actual game clock 5 seconds + 0.05 seconds per move, concurrency 2. Both sets
run to their declared size without tuning/stopping based on intermediate score.
The first set is not used to select another candidate for the second set.
A public-start dummy search loads networks outside the game clock, followed by
ucinewgame to clear game state. Every actual search records elapsed time; the
referee charges its measured time to the game clock. A two-game clock smoke
finished 1 win / 1 loss, no technical errors, mean move times about 137 ms
(candidate) and 140 ms (baseline). This smoke is not a confirmation result.

One training seed does not meet the separate multi-seed training requirement;
these runs assess whether this frozen candidate warrants that larger test.
