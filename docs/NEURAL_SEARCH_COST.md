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
