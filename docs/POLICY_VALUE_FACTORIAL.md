# Separate complete policy and value functions

The original first neural RL update fit search targets but lost its fixed-clock
warm-control comparison24W/1D/39L. That alone does not identify whether the
policy function, value function, their interaction, or search throughput causes
the regression. A later LeafBatch8-trained checkpoint is a different experiment
and is not substituted into this diagnostic after observing its matches.

We freeze the original warm model and the first LeafBatch32 RL model. Their
SHA256digests start4ff3c675 and168ae3d9, respectively. No parameters are fitted
or copied between networks. In particular, each trunk, policy head, value head
and all batch-normalization statistics stay with its own checkpoint. The wrapper
returns policy logits from one complete forward and value from another.

| Candidate policy | Candidate value | Control policy/value |
| --- | --- | --- |
| RL | Warm | Warm/Warm |
| Warm | RL | Warm/Warm |
| RL | RL | Warm/Warm |

Both players always execute **two complete forwards** per leaf batch, including
identical Warm/Warm or RL/RL checkpoints. Nothing is cached or skipped for these
identical combinations. This keeps the same architecture and forward-count
recipe on both sides, although behavior-dependent batch sizes, simulations,
wall-time variation and FLOPs can still differ. The diagnostic is under actual
10+0.1clocks, not an exact fixed-simulation causal decomposition. It cannot be
pooled with the original one-forward match to estimate an additional effect.

All three64-game contrasts use the same new32paired physical layouts,
seed9580001, LeafBatch32, one CPU thread per actor, two concurrent games and
one shared dedicated GPU. Reusing layouts permits subsequent paired method
comparisons but makes the three screens correlated. They are not independent
replications. The plan fixes all contrasts before outcomes and runs them all
regardless of intermediate results. No teacher gate, promotion, training seed
sweep or checkpoint selection is involved.

Two targeted tests verify complete function outputs and untouched parameters/
normalization, then bitwise search equality for identical checkpoints and
isolation from poisoned opponent-private ledgers. Four real-checkpoint CPU
protocol checks cover all combinations at16completed simulations, including a
private dark capture. These establish integration, not playing strength.

The study waits for the timed-halving campaign to finish before using its GPU.
Each contrast receives independent native legal/private-history/terminal/clock
replay and raw-game archiving. Source/model hashes are checked while queued and
before and after every stage. Current evidence:
`results/policy-value-factorial-plan.json` and
`results/20261027-factorial-protocol-smoke.json`.

The first contrast, RL-policy/Warm-value versus Warm-policy/Warm-value, has
completed23W/0D/41L:35.9375%score, paired bootstrap95%[25%,48.4375%]. All3,007
moves pass native seeded-identity, legality, full-history terminal and actual-
clock replay, with no technical games or ambiguous margins. Retaining the warm
value function therefore does not rescue the learned policy in this diagnostic.
This is a negative policy-function signal under the matched two-forward clock
profile; the interval is per-contrast and unadjusted for the three planned
comparisons. It is not a completed factorial analysis, an equal-simulation
decomposition or proof that the value function is harmless. The two remaining
contrasts continue without changing the plan.

Evidence: `results/20261027-factorial-rl-policy-warm-value{,-audit}.json`.

The second contrast, Warm-policy/RL-value, completed24W/2D/38L:39.0625%score,
paired bootstrap95%[27.34375%,50.78125%]. All2,949moves/clocks are clean under
the same independent replay. This also gives no gain while preserving the
warm policy, though its interval includes parity. The final RL-policy/RL-value
contrast continues on the same original layouts and settings; conclusions
about cross-function effects wait for its completion. Evidence:
`results/20261027-factorial-warm-policy-rl-value{,-audit}.json`.
