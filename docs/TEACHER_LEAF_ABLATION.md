# Separate neural value approximation from information-set search

The warm CNN policy/value PUCT remains far below the native jqe teacher. Three
small-data neural RL iterations have not improved their preceding actors on
development screens. More search during self-play is being tested separately.
A second diagnostic asks whether replacing approximate CNN values with the
teacher's static evaluator improves the existing search at fixed clocks.

`research/teacher_leaf.py` retains the original frozen CNN policy, observer
encoding, root-consistent particles, chance transitions, private observations,
PUCT selection, virtual_loss3 and completed-batch deadlines. Only the scalar
nonterminal leaf value is replaced by tanh(cp/860), with cp from the original
quantized it014 NNUE. The fixed860 scale matches the existing pretraining
score transform; it is not selected from match outcomes. There is no fitting,
new learned evaluator, native PVS search or quiescence in this substitution.
Terminal rule outcomes continue to override static values.

The C ABI in `research/nnue_observer_bridge.cpp` compiles unchanged MIT jqe NNUE,
position and table sources. It accepts an actor-observation FEN: public board,
actor side, actor-known inventories and no-capture count. It never accepts a
true deal or another actor's private ledger. Unknown own captured types stay
in the known upper-bound inventory as in jqe's original observer representation.
The root and simulated actor private records are produced by the unchanged
original information-set search. The evaluator remains history-blind; complete
rule history is still retained in the search and terminal checks.

`scripts/build_nnue_observer.py` writes a relative-path manifest with source,
library hashes and compiler flags. An isolated process is required for different
native NNUE weights. Two targeted tests verify exact centipawn agreement with
the unchanged original binary on seven observer views spanning both colors,
private captures and a fully revealed board; legal normalized fixed-simulation
search and independence from poisoned actual opponent private slots. A separate
actual warm-model CPU UCI smoke completes8visits after a private dark capture.
These checks prove integration behavior, not playing strength.

## Fixed development screens

`results/teacher-leaf-screen-plan.json` fixes two64-game paired-layout screens,
actual10+0.1, one CPU thread each, two games concurrent:

| Candidate | Opponent | Layout seed |
| --- | --- | --- |
| Warm CNN policy + teacher static value, LeafBatch8 | Original warm CNN policy/value, LeafBatch8 | 9230001 |
| Same hybrid | Unchanged original jqe+it014 NNUE/PVS | 9240001 |

Both neural actors use the same dedicated GPU. Native static NNUE executes
serially in the candidate's process; its CPU cost is included in measured wall
time. The native teacher opponent uses CPU only. No equal total compute/FLOP or
Hash-memory claim is made. The hybrid's CNN value head is still computed before
its output is replaced, so this first diagnostic does not optimize that small
extra cost away.

The experiment is queued after the fixed LeafBatch confirmation campaign,
regardless of that campaign's results, to avoid sharing its match GPU. Sources,
library and models are checked before/after every stage. Independent seeded
native identity/legal/terminal/clock replay and raw-result archiving follow
both screens. There is no adaptive early stopping or result-selected checkpoint.
The dependency campaign has completed and this experiment is now running its
warm-model comparison on the released GPU. That first64-game comparison has
now completed37W/1D/26L,58.59375%score, paired bootstrap95%
[47.65625%,69.53125%]. All2,930moves/clocks replay cleanly. The interval includes
50%, so the positive point estimate is not established superiority. The fixed
teacher comparison is still running; no promotion or learned-model gain.
First report/audit: `results/20261026-teacher-leaf-vs-warm{,-audit}.json`.
Any future gain is attributable in part to the unchanged upstream teacher
value function, not a newly learned research contribution. A fully public-only
value substitution would be a separate future ablation if evidence warrants it.
