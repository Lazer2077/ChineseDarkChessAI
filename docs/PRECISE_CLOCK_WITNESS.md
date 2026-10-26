# Supplemental high-resolution wall-clock witness

The first opponent-only anchor screen, AB-JChess versus the native classical
Pika WASM branch, completed 40 games at 23W/0D/17L for AB. Its raw game records
round move time to milliseconds. The original independent replay therefore
reports 22 ambiguous lower-bank checks, minimum -24.5ms, although no engine
time-forfeited and all 2,605 moves were legal and correctly adjudicated.
Those original flags and the frozen screening controller's ineligibility remain
unchanged. The following is explicitly a supplemental post-match witness, not a
retroactive pass of that controller's original rounded-audit condition.

## A stricter interval already exists

The frozen `scripts/referee_entry.py` wrapper starts `monotonic_ns` before calling
the original `Engine.go` and records elapsed time after it returns, before log
I/O. The original charged wall interval starts and ends inside that call. The
wrapper therefore measures an upper duration that also includes token preparation
excluded by the inner timer. Warmup is identified separately, before game clocks.
Neither actor uses a different timing allowance in this witness.

For each fresh engine PID, with initial bank B, increment I and outer durations
e_i, compute:

```math
L_i=B+(i-1)I-\sum_{j=1}^{i}(e_j+\epsilon),\qquad \epsilon=0.001\text{ ms}.
```

The guard exceeds the floating/nanosecond conversion resolution. Increments are
only added after checking the preceding move. If every L_i is nonnegative, every
contained engine interval fits the original cumulative wall-clock bank. A negative
outer bound remains inconclusive; it is not itself proof of a time violation.
No time grace, altered adjudication or rounded-time tolerance is introduced.

PIDs need not be paired with particular layout seeds: the claim holds for every
logged engine call. The audit checks one warmup per PID, exactly two fresh engine
lifetimes per game, consistent labels, sequential per-PID plies starting at 0/1,
clock/increment fields, and equality of the aggregate (engine, ply) call counts
with all recorded game moves. The unchanged referee creates new processes per
game; repeated warmups/PID reuse, missing calls and random openings are rejected.
It verifies frozen source hashes, the known wrapper digest and the unchanged
referee checkout. It refuses failed, void, incomplete or technically unclean runs.
Legality/seeded-identity/terminal replay remains a separate requirement.

Implementation: `scripts/audit_outer_clocks.py`; four tests cover positive and
negative bounds, the guard, duplicate/missing/reused lifetimes and increment order.
The witness is sufficient, not necessary, for wall-clock safety and does not
estimate strength. Future acceptance plans may explicitly use this sufficient
certificate in addition to rounded replay; existing frozen plans are not edited.

## First supplementary result

`results/20261026-baseline-screen-abj-vs-pika-wasm-native-outer-clock-audit.json`
certifies all 2,605 calls in 80 fresh engine processes, with minimum conservative
outer-bank lower bound **2.02679ms**. The content-addressed compressed timing log
is archived alongside its digest. This resolves the wall-clock question more
precisely, while retaining the original rounded-audit outcome. The 57.5% score's
paired bootstrap interval [42.5%,72.5%] still gives no clear ordering, independently
of timing. No baseline is eliminated or credited as defeated by our model here.
