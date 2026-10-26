# Sharing search decisions after the last dark piece

This experiment asks whether declining hidden-state relevance can reduce
duplicate search statistics, beyond canonicalizing inputs to a neural evaluator.
It uses the pinned Yuan information-keyed PUCT and the same canonical observer
encoder as the residual model. It is an implemented mechanism experiment, not a
confirmed strength improvement.

## Conditional decision abstraction

The original search keys decisions by actor inventory, public simulated history
and captor-private observation history. Once all on-board dark pieces disappear,
different sampled private capture paths may describe the same future public game.
We test sharing those decisions while retaining:

- Side to move and complete public board.
- The entire simulated public move/reveal path from the current search root.
- No-capture count, consecutive checks, check anchors and every chase field.
- The complete public position-count map and the actual rules configuration.

Only dead inventory and private capture paths are omitted from the new key.
Still-dark pieces never enter the abstraction, even if an inventory implies their
identity. Missing/incomplete repetition memory also disables it. Search uses
actual Jieqi rules and never converts the continuation into ordinary Xiangqi.
Keeping the public path avoids cyclic decision sharing within a simulation.

The shared neural policy/value evaluator already has identical canonical inputs
for these dead-inventory variations. The phase value MLP is not changed by this
experiment. Chance probabilities, root particle sampling, legal moves and backup
counts remain inherited. The source checkout and other running experiments stay
unchanged; our subclass patches only decision-key calls and the existing
completed-batch deadline into a runtime copy of the original search function.

This is an abstraction for a public perfect-information continuation value. An
arbitrary fixed opponent may still choose moves based on its dead private history;
forgetting that information need not preserve an exploitive best response to that
particular policy. We do not claim equivalence of all behavioral strategies, a
precise strategy-conditioned posterior or a new equilibrium convergence theorem.

## Fixed-corpus diagnostic

The corpus is the 12 training episodes of the completed neural self-play pilot,
excluding its four checkpoint-selection games. The fixed rule selects seven
states with one remaining own dark piece and a non-singleton own inventory, plus
two opening and two fully public controls. These positions are correlated, not
independent strength samples. Exact moves, observations and seeds are certified
by complete episode replay.

Both instrumented original-key and public-sharing variants use the same warm-start
model, 1024 completed simulations, leaf batch 16 and two repetitions with alternating
order. The selected corpus is saved before measured searches. Diagnostics count
distinct original full-public keys, continuation equivalence classes, potentially
removable private contexts, actual leaf evaluations, root visits and elapsed time.
CUDA/model startup is warmed separately. Timings remain exploratory; no per-node
count or speedup is treated as a win or independent timing confidence interval.

The unshared control reproduces original fixed-simulation actions/root statistics.
Tests independently exercise dead-inventory invariance, public history and each
rule-state field, refusal of still-dark/incomplete-history abstraction, actual
root visit normalization and own-versus-opponent inventory corpus selection.

Attempt 01 failed during corpus selection because a NumPy record retained a
singleton dimension. It completed zero measured searches; source, frozen plan
and failure evidence are preserved. The corrected attempt uses a fresh directory
and plan. Evidence: `results/20261026-public-continuation-preflight-failure.json`;
current recipe: `results/public-continuation-profile-v2-plan.json`.

This idea builds on known information abstraction and the already recognized
irrelevance of off-board hidden status in an all-revealed continuation; see
[the research baseline audit](RESEARCH_BASELINES.md). Novelty and strength must be
established through evidence, rather than asserted from the phase gate itself.


## Completed mechanism diagnostic

The corrected run completed 44 searches / 45,056 simulations over 11 correlated
states. All seven last-own-dark positions came from one episode. No removable
private-context duplicate was encountered in this corpus at 1024 simulations.
For every original/public-key pair, actions, root values and unique leaf counts
were identical. No node savings or strength gain is established. Instrumentation
constructs public equivalence classes in both modes, so these elapsed times do
not compare with an uninstrumented production baseline.

A separate legal five-ply fixture captures two still-dark pieces in different
private identity orders. Both branches have identical public paths, rule memory,
legal moves and nonterminal outcomes, but different original private keys; the
public key merges them only after all dark pieces disappear. This demonstrates
a real semantic alias, rather than a speed benefit in the measured games.

With a fully public search root, no new private capture paths arise. Preserving
its complete public path therefore leaves little opportunity for this particular
key abstraction. Any benefit would have to come from private-context alternatives
inside a previously partial root. The diagnostic does not justify launching a
large strength match or dropping public history to manufacture more sharing.
Evidence: `results/20261026-public-continuation-profile.json` and 14 correctness
checks. Further hypotheses should target measured sources of search cost.
