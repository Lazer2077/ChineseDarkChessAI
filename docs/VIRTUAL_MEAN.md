# Virtual mean with observer-specific chance search

The first neural RL iteration lost to its previous actor and remains far below
the unchanged native teacher. Root coverage and three-root RNG diagnostics do
not isolate the cause. We therefore test one inherited batching mechanism before
expanding training: [Cazenave, Batch Monte Carlo Tree Search](https://arxiv.org/abs/2104.04278)
motivates preserving a node's current mean while leaf evaluation is pending.

The existing actor increments a selected node's visit count and temporarily adds
3 to its value sum. Our experimental rule instead reserves one visit with credit
m = W/N (zero if N=0), giving N'=N+1 and W'=W+m. When the real leaf value v is
backed up, replace that specific credit by adding v-m. The root records a real
visit only on backup, as in the inherited search. The reservation principle is
attributed to prior work; no new policy-improvement theorem is claimed.

## Private information and chance-chain correctness

`research/virtual_mean.py` records credits per simulation path rather than in a
node FIFO. A terminal path may finish before an earlier pending path. Shared
information-set nodes must therefore remove the right path's credit even when
completion order changes. Moving-piece and captured-piece reveal layers keep
their own player perspectives: value changes sign only when adjacent path nodes
have different players, exactly as in the inherited backup.

Determinization, without-replacement reveals, actor-private inventory, information
keys and complete rule/public/private histories remain in the original search.
The new rule changes selection statistics, so it need not reproduce fixed-budget
batched actions. Three tests check credit conservation with out-of-order backups,
same-player chance layers, batch1 agreement with the original within floating
rounding, and a complete real self-play game whose visits, observations, hidden
reveals and terminal labels pass independent seeded replay. Combined with outer-
clock and original neural-search checks, 10 relevant tests pass.

The transformed runtime search source replaces exactly four reservation sites,
uses a guarded path initializer, and keeps the existing completed-batch deadline.
Its digest is exposed by the separate UCI actor. The original source/classes and
currently frozen actors are not modified. This is a research implementation of
one pending-value mechanism, not reproduction of every mechanism in the paper.

## Fixed queued screen

`results/neural-virtual-mean-screen-plan.json` fixes a 64-game actual10+0.1
development comparison against scalar virtual loss3, seed8760001. Both actors
use the same original selected shared warm-start model, leaf batch32, one CPU
thread each and the same GPU, two games concurrent. No training, checkpoint
selection, private-information access or architecture differs between arms.

The screen waits until the separate LeafBatch8-versus32 experiment fully
completes; it does not share that GPU while those games run. The same original
checkpoint is used regardless of the preceding result. Frozen source/model
hashes, complete paired games, seeded legal/identity/terminal replay and clock
audit are required. A failed/missing dependency stops the queue instead of
restarting it. The dependency completed, and the virtual-mean screen is now running, with no
strength or promotion claim. Its
result must precede any use of virtual mean in a future self-play training run.

The fixed screen completed38W/0D/26L (59.375%score), paired bootstrap95%
[50.0%,68.75%], conservative Hoeffding95% [35.36693%,83.38307%]. All3,602moves
and clocks replayed cleanly, with no technical games, voids or rounding ambiguity.
This does not establish promotion or a teacher gain. Evidence:
`results/20261026-neural-virtual-mean-screen.json` and its `-audit.json`.
The next queued LeafBatch8 confirmation has started as planned regardless of
this result, using the original checkpoint rather than this variant.
