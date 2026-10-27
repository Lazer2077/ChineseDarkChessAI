# Separate target entropy, imitation error and playing strength

LeafBatch8generation produced a lower own-selection policy loss than the
original LeafBatch32generation. Both use the same initial actor,256completed
simulations and first-iteration environment/training seeds, but their actions
and trajectories differ. Raw cross-entropy cannot distinguish a simpler target
distribution from a model becoming better at matching that distribution.

`results/visit-target-diagnostic-plan.json` fixes the existing32selection episodes
(indices256–287) from each arm and three frozen models: warm, original RL1, and
the LeafBatch8 RL update. It evaluates every model on both splits with CPU FP32,
one thread, original float64visit targets and the original full2550-action
softmax. There is no fitting or new test-set selection. The own selected models'
metrics reproduce archived GPU selection losses within3e-4. These episodes
already selected checkpoints; the matrix is diagnostic, not independent
generalization or playing-strength evidence.

For each distribution, CE(target,model)=H(target)+KL(target||model). These full-
action training KLs include mass assigned outside legal actions; they are not
KLs between the search's legal-normalized policies.

| Selection trajectories | Model | Target entropy | Policy CE | Policy KL | Terminal MSE |
| --- | --- | --- | --- | --- | --- |
| Leaf32 | Warm | 3.059171 | 5.574549 | 2.515378 | 0.759373 |
| Leaf32 | RL-Leaf32 | 3.059171 | 3.877386 | 0.818215 | 0.760621 |
| Leaf32 | RL-Leaf8 | 3.059171 | 4.341925 | 1.282754 | 0.769807 |
| Leaf8 | Warm | 1.945806 | 2.887939 | 0.942133 | 0.756529 |
| Leaf8 | RL-Leaf32 | 1.945806 | 2.814459 | 0.868653 | 0.673833 |
| Leaf8 | RL-Leaf8 | 1.945806 | 2.614054 | 0.668248 | 0.671377 |

The own-model CE gap1.263332nats decomposes into1.113365nats of target-entropy
difference and0.149967nats of KL difference: about88.13%of the numeric gap is
the entropy term. This is an algebraic description of different datasets and
models, not a causal claim about what batch size alone changes. Within each
fixed dataset, the matching model best fits that dataset's search targets; the
old original RL update nevertheless lost to the warm actor in actual games.
Better visit imitation is not sufficient evidence of policy improvement.
The batch8 actor's separately frozen fresh warm match has now completed
29W/1D/34L,46.09375%score, bootstrap95% [35.9375%,56.25%], with clean native
replay and clocks. Better own-target fitting has not established stronger play.
Its teacher screen completed0W/1D/63L, also clean; no weights are promoted.
Both finished matches are summarized in
`results/20261027-neural-small-batch-rl-summary.json`.

A separate fixed greedy-policy-versus-PUCT match now checks the missing
planning-versus-prior control, with the same original warm network and actual
10+0.1clocks. One actor uses one legal-normalized policy inference and zero
tree simulations, while the original PUCT actor uses its stronger LeafBatch8
configuration and may use its full time. Both have one CPU thread and the same
GPU; this is not equal FLOPs or equal work. Greedy ignores the computed value
head and never determinizes/selects/backs up a search tree. It retains the
original observer reconstruction, legal mask and normalization/tie order.
Two tests verify zero visits, exact masked-prior choice and independence from
the actual opponent private ledger. An actual warm-model CPU UCI smoke reports
zero simulations explicitly. The64paired-game development recipe is
`results/policy-only-screen-plan.json`, seed9530001. No weights change or
result-selected stopping is allowed. Native identity/legal/terminal/clock
replay follows completion. The contrast is running, without a conclusion that
planning improves or harms the original policy.

Full matrix and dataset statistics:
`results/20261027-visit-target-diagnostic.json`;
algebraic decomposition: `results/20261027-visit-target-summary.json`.

## Observer-private legality boundary audit

The original Yuan long-check/chase forbidden-move path can simulate a revealed
piece when deciding which actions to filter. This raises a concrete question:
can an action mask vary with an own remaining inventory the actor does not know?
An offline audit checks actual completed training roots before attributing the
RL failures to this mechanism. It does not change rules or running jobs.

`results/observer-legal-mask-audit-plan.json` fixes all288episodes of both arms,
576games /33,040positions under the pinned Yuan/Cython rules profile. Every
episode is seeded-replayed. At every actual ban root with live own dark pieces
and unknown own captured types, all compatible captured-count vectors are
enumerated, rather than just testing a few random worlds. Each hypothetical
world retains the same actor information state, public board, complete rule
history and counters; only the hidden own live inventory changes.

| Arm | Actual ban roots | Ban roots with uncertain live inventory | Compatible worlds checked | Sensitive masks |
| --- | --- | --- | --- | --- |
| Leaf32 | 10 | 8 | 72 | 0 |
| Leaf8 | 11 | 7 | 34 | 0 |

No mask difference or selected action outside the common legal intersection
was observed at these15eligible roots. There is therefore no observed evidence
that this mechanism explains the losses of these two generation arms. The
coverage is limited: this is not a proof over other reachable states, future
episodes, counterfactual interior search nodes, or other rule profiles. Native
external benchmark rules and the original Yuan training profile remain distinct.
Full counts and exhaustive-audit recipe:
`results/20261027-observer-legal-mask-audit.json`.
