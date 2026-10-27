# Timed root-halving prefix with survivor continuation

Fixed-work root halving has a positive64-game development result against the
same warm CNN/PUCT control, but it consumes exactly256simulations regardless
of the available time. Its completed independent200-game confirmation is reported separately.
A timed actor needs a defined recommendation if interrupted early and useful
continuation if the prefix finishes early. This new hybrid tests those choices
under the original10+0.1actual-clock benchmark stratum.

## Defined hybrid

`research/anytime_halving.py` retains the original observer encoder, root-
consistent particles, interior PUCT, chance transitions, private observations,
history and value backups. It uses zero-noise root halving over at most16prior-
selected actions for a planned256simulation prefix. The root-Q normalization
and ranking are the same as the [fixed-work prototype](ROOT_HALVING.md).

After that complete prefix, the final two candidates remain available. The
same search tree continues with PUCT restricted to them. Original priors are
renormalized over survivors, and the PUCT exploration term uses their visit
mass. Rejected moves are not reopened. Thus extra time refines the remaining
choices without rebuilding the tree, restarting rollouts or committing forever
to one winner. This is an explicit new hybrid ablation, not the full Gumbel
AlphaZero algorithm, an optimal anytime scheme, or a policy-improvement proof.

The original completed-batch deadline check remains: after at least one batch,
stop between completed leaf batches. Batches cannot cross a halving boundary.
Recommendation uses the most visited root-action set, breaking ties by the
same prior/scaled-Q score. It works for incomplete prefixes. Temporary virtual
values must be removed before any elimination/recommendation. The actor returns
actual normalized root visits and completed count, not a fabricated fixed256.
All external wall time remains charged, with the original20msmove-overhead
budget and no additional clock grace. A5msrequest can take longer than5ms to
finish its first batch; it is not a hard real-time latency promise.

## Prefix rounding correction and verification

The first prototype is retained at commit`6eb59a0`, with its CPU integration
records. Extra tests found that five legal candidates can leave a one-simulation
rounding remainder: entering the survivor phase at255would violate exact prefix
equivalence. The corrected implementation completes the fixed allocator's
remaining one-winner work before restoring the last two candidates. No frozen
timed strength study uses the earlier source.

Twenty-one targeted tests now cover every candidate count1through16, matching
all allocations/visit/value sums/recommendations of the fixed256prefix; real
particle-search equality at256; reuse beyond the prefix; conditioned survivor
priors; expired deadlines; pending-credit rejection and actual opponent private-
ledger poisoning. Together with fixed-halving/original-search tests,32checks
pass. No original source or running experiment is changed.

An actual warm-network CPU UCI test completes384simulations with survivor
continuation beginning at256, then returns a complete8leaf batch for a5ms
request. Corrected-source two-game CPU integration uses actual10+0.1and has
1W/0D/1L,126moves/clocks independently clean. These are integration checks,
not statistical playing-strength evidence. The earlier74-move smoke and its
preliminary source remain archived separately.

Evidence: `results/20261027-anytime-halving-protocol-v2.json` and
`results/20261027-anytime-halving-cpu-v2{,-audit}.json`.

## Frozen timed development study

`results/anytime-halving-screen-plan.json` fixes two fresh64-game paired
development screens from the original warm checkpoint:

| Candidate | Opponent | Layout seed |
| --- | --- | --- |
| Timed hybrid,LeafBatch8 | Original warm PUCT,LeafBatch8 | 9560001 |
| Same hybrid | Unchanged native jqe+it014 teacher | 9561001 |

Both use actual10+0.1, one CPU thread per actor and two concurrent games. Neural
actors share one dedicated GPU; the native teacher is CPU-only. Equal clock
allowance does not imply equal simulations, model calls, FLOPs or Hash memory.
The original prefix/zero-noise/16candidate/two-survivor choices and source/model
digests are fixed before outcomes. No fitting, new training seed, result-selected
checkpoint, optional stopping or between-screen retuning occurs. Both contrasts
finish regardless of the first result, followed by native legal/private-history/
terminal/clock audits and raw archiving. The aligned-weight campaign completed,
and this study has started its warm-model comparison on the released GPU.
No superiority or promotion
is implied by passing tests or producing this timed actor.

The first frozen timed contrast has completed36W/2D/26L against warm PUCT:
56.25%wins,57.8125%score, paired bootstrap95% score interval
[46.09375%,69.53125%]. This interval includes parity, so improvement is not
established. All3,614moves pass native seeded-identity/legal/terminal/actual-
clock replay; there are no technical games or ambiguous clock margins.
The unchanged native-teacher contrast continues with its original frozen
settings. No tuning follows the warm-control result. Evidence:
`results/20261027-anytime-halving-vs-warm{,-audit}.json`.
