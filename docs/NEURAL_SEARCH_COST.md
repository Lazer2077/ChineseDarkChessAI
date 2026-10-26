# Neural search cost: diagnostic evidence

The first completed on-policy iteration is still weak: 24W/1D/39L versus its
previous neural actor and 1W/0D/63L versus the native teacher, actual 10+0.1.
Independent move/clock audits are linked in `NEURAL_SELFPLAY.md`. Expanding the
model is an available experiment, but does not by itself address search cost.

`results/20261026-neural-root-profile.json` records one fully hidden initial
root at seed 20261031, the original selected shared residual warm-start checkpoint,
one CPU thread plus GPU, 32-leaf batches and 1,024 completed simulations.
With cProfile enabled, wall time was 3.109s, node expansion cumulative time
1.244s and the complete neural evaluation path 0.631s. Python neural-forward
cumulative time was 0.296s; this is not an isolated GPU-kernel benchmark.
Cumulative times overlap and must not be summed. One root and profiler overhead
cannot establish general throughput or a playing-strength bottleneck on every
phase. It does identify allocation/expansion as worth further inspection.

## Existing lazy allocation: counterfactual, not a new gain

The pinned upstream `AZConfig.lazy_chance_allocation` default is **already true**
in our current actor. A planned separate actor was canceled before any match
after a targeted test exposed an incorrect assumption that this default was
false. No new implementation, optimization or win-rate gain is claimed.

The separate diagnostic `results/20261026-lazy-chance-diagnostic.json` explicitly
sets eager false versus lazy true on the same root, with three alternating-order
repeats, 1,024 completed simulations each, 6,144 measured simulations total.
Every selected move, root policy, child visits/value sums/priors and root value
sum matched exactly. Reachable nodes were 359,148 eager versus 48,700 lazy.
The median eager/lazy wall ratio was 1.5205 (individual ratios 1.3826, 1.5205,
1.9757). This compares a deliberately eager **counterfactual** with the existing
default; it does not speed up the default actor. External load, GPU scheduling
and one correlated root limit timing generalization. The relevant implementation
belongs to upstream, not this project.

`tests/test_lazy_allocation_control.py` checks the actual default and fixed-
simulation equivalence independently with a small CPU model, preventing the
counterfactual from being mistaken for an improvement. The initial test failed
because it expected the default to be false; that assumption was corrected and
the redundant actor/match plan removed before launch. Reproduce the diagnostic
using `scripts/profile_neural_root.py` and `scripts/compare_lazy_chance.py`.

Further cost work must compare with the actual default, retain complete observer
and rule histories, and use representative partial and fully public roots.
Changes to allocation alone need fixed-simulation equivalence before actual-clock
strength screens; alternate priors or information sharing require separate
algorithmic ablations rather than being labeled equivalent optimizations.

## Original optional compiled expansion: measured against the actual default

The existing actor uses Python node expansion (`JIEQI_SEARCH_BACKEND` defaults
to `python`), independently of its enabled lazy-allocation default. Upstream
also supplies an optional Cython expansion method. It was built without edits
in a separate copy; original sources and all frozen running campaigns remain
unchanged. Build provenance: `results/20261026-yuan-search-build.json` and
`scripts/build_yuan_search.py`. This is inherited implementation work, not a
new search algorithm.

The fixed `results/search-backend-control-plan.json` selects an opening root,
one uncertain-last-own-dark root and one fully public root from certified pilot
TRAINING episodes. Both backends use the actual lazy default, identical warm-start
weights, private/rule histories, seeds, 32-leaf batches and 1,024 complete
simulations. Three alternating-order repetitions in separate processes yield
18 searches / 18,432 measured simulations. All nine Python/Cython pairs match
exactly in root action, full policy, visits, value sum, child visits/value sums
and priors. Existing fixed-simulation/deadline/private-observer tests also pass
with the compiled backend (3 tests).

Median Python/Cython wall ratios by root are 1.0298 opening, 1.0621 last-dark,
and 0.9929 fully public. These small exploratory differences across three
correlated states do not establish a broad runtime or strength improvement.
The Python node constructor remains unchanged by upstream's optional backend.
We have not enabled it in existing training, promoted it, or started a redundant
large strength campaign from these timings alone. Raw root fingerprints and
samples: `results/20261026-search-backend-control.json`.

## Root coverage: testing a low-simulation explanation

[Policy Improvement by Planning with Gumbel, Danihelka et al., ICLR 2022](https://openreview.net/pdf?id=bERaNdoegnO)
identifies policy-improvement problems when root actions receive no visits and
studies without-replacement sampling and sequential halving. This is relevant
literature, but its correct-action-value assumption is not established for our
approximate private-information search. We have not implemented or claimed its
policy-improvement guarantee in Jieqi.

Before launching another search variant, `scripts/diagnose_root_coverage.py`
replayed only the first iteration's 256 TRAINING games (13,924 positions),
excluding its 32 checkpoint-selection games and all strength-test layouts.
Every episode digest, observer/visit/terminal audit and legal-action support is
checked. The actual targets contain 256 complete root visits per position.
Position-weighted means: 32.366 legal actions, 29.778 visited actions, 94.124%
legal-action coverage; 22.278% of positions leave at least one legal action
unvisited. Equal-game means are 93.655% coverage and 22.917% with an unvisited
action. Effective visit actions average 24.016, and maximum visit mass 19.685%.
Counts and partial/full phase breakdowns:
`results/20261026-neural-root-coverage.json`.

These observations weaken an explanation based on most legal actions being
unvisited during this fixed-simulation training generation. They do not measure
short-clock test roots, chance-conditioned Q accuracy or actual policy
improvement. Visit entropy includes root Dirichlet exploration noise; a diffuse
label is not itself an error. The next search-target study should assess value
estimation and chance-sampling reliability, and distinguish these from action
coverage, before attributing losses to root allocation or adopting Gumbel
planning. This is a diagnosis, not a newly validated algorithm.

## Search RNG stability across information phases

The fixed `results/neural-search-stability-plan.json` repeats the same three
certified TRAINING roots, with eight search RNG seeds and budgets 256/1,024,
no Dirichlet noise and temperature0. All 48 searches / 30,720 simulations
completed. Root action choice was identical across seeds within each root/budget.
The fully public root's entire policy, child Q/visits and root value were exactly
seed-invariant, consistent with declining future reveal uncertainty. This is
one observed control, not a theorem about all games or opponent private policies.

| Root | Budget | Root-value sample SD | Mean pairwise policy total variation |
| --- | --- | --- | --- |
| Opening | 256 | 0.005864 | 0.135324 |
| Opening | 1,024 | 0.002706 | 0.128104 |
| Last uncertain own dark | 256 | 0.000401 | 0 |
| Last uncertain own dark | 1,024 | 0.000154 | 0.011440 |
| Fully public | 256 | 0 | 0 |
| Fully public | 1,024 | 0 | 0 |

The last-dark root has near-saturated positive values and is not a representative
hard decision. Last-dark and public roots choose different actions between the
two budgets despite stability within each budget: absence of RNG variance does
not imply sufficient search depth or correctness. Adaptive Q variation is not
an unbiased expected-return estimate or an independent confidence interval.
Three correlated roots cannot isolate the cause of poor match strength.
Full samples and action statistics: `results/20261026-neural-search-stability.json`.

## Fixed leaf-batch ablation

[Cazenave, Batch Monte Carlo Tree Search](https://arxiv.org/abs/2104.04278)
studies the tradeoff between batched inference and pending tree evaluations,
including virtual-loss/virtual-mean mechanisms. Our inherited default virtual
loss is 3 and GPU actor batches are 32. With 256 simulations and about 30 visited
root actions, there may be a meaningful cost to pending, not-yet-backed-up leaf
values. This is a hypothesis, not an established failure or a transferred Go result.

A fixed 64-game actual 10+0.1 development screen now compares LeafBatch8 against32
using the identical warm-start checkpoint, original Python/lazy search, one CPU
thread each and the same GPU with two concurrent games. Smaller batches provide
more frequent feedback but may complete fewer simulations under the clock.
Only leaf batch differs; no new training or architecture is involved. Seed8750001
is separate from prior screens; legal/private-history/actual-clock replay and
source/model checks follow completion. Recipe:
`results/neural-leaf-batch-screen-plan.json`. The screen completed at
38W/7D/19L (64.84375% score), paired bootstrap95% [54.6875%,74.21875%],
conservative Hoeffding95% [40.83568%,88.85182%]. All3,472 moves/clocks replayed
cleanly, with no technical games, voids or rounding ambiguities; minimum
conservative remaining clock553.5ms. This is a positive development screen,
not promotion or a demonstrated gain against the native teacher. Evidence:
`results/20261026-neural-leaf-batch-screen.json` and its `-audit.json`.
Existing frozen self-play/width experiments remain unchanged.

A separate [virtual-mean adaptation](VIRTUAL_MEAN.md) completed38W/0D/26L in its
fixed64-game development screen, bootstrap95% [50%,68.75%]. This changes pending
value accounting, not model size or the observation contract; the result is
inconclusive and does not support adoption over scalar virtual loss.

The positive LeafBatch8 development result now has a fixed follow-up:
`results/neural-leaf-batch-confirmation-plan.json` schedules one fresh200-game
confirmation against32 (seed8780001), then a separate64-game unchanged-teacher
diagnostic (seed8790001), both actual10+0.1. It starts after virtual mean finishes
regardless of that result, on the same GPU, with the same original checkpoint.
Neither screen is selected or retuned from virtual-mean outcomes.

The200-game independent LeafBatch8 confirmation completed131W/7D/62L,
67.25%score, bootstrap95% [61.5%,73%] and conservative Hoeffding95%
[53.66898%,80.83102%]. All11,024moves and actual-clock checks replay cleanly,
no technical results/voids/rounding ambiguities, minimum lower clock581.5ms.
This independently supports a fixed-clock search-configuration gain for this
specific unchanged warm network. It is not a newly learned-model gain or an
advantage over the teacher. The fixed64-game teacher diagnostic is still live.
Evidence: `results/20261026-leaf-batch8-confirmation{,-audit}.json`.

The original three low-simulation RL updates are now closed and negative; a
new one-iteration training control uses the same warm model, first-iteration
environment/training seeds,256+32episode split,256simulations,4epochs/batch128/
lr1e-5 and original normalization, changing only generation LeafBatch32to8.
Evaluation deliberately stays LeafBatch32 for both neural actors, so a training
effect is not confused with the already established inference configuration
effect. Trajectories, targets, terminal outcomes and optimizer step count may
change despite matching seeds. Recipe: `results/neural-small-batch-rl-plan.json`.
It runs on a GPU released by the completed original RL series, separately from
the2048-simulation/LeafBatch32 search-effort control. Both fixed warm/teacher
development screens complete regardless of outcomes, followed by independent
native audits before any strength interpretation. No promotion is implied.
