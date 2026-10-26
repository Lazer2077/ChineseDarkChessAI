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
