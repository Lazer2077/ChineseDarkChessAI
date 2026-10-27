# Expert supervision on learner-generated, observer-limited histories

The current warm policy was fitted to strong-teacher trajectories. Subsequent
self-play visits and outcomes come from a much weaker learner, and both its
policy and value functions regress. Constant warm-policy anchoring limits some
regression but does not outperform the original warm actor. This motivates a
different question: can the unchanged native expert provide useful policy
labels at states actually reached by the learner?

This hypothesis follows the distribution-shift motivation of DAgger
([Ross, Gordon and Bagnell, AISTATS2011](https://proceedings.mlr.press/v15/ross11a.html)).
It is not a new imitation-learning algorithm or a claim that the paper's
no-regret guarantees apply to our compressed POMDP observation encoder,
changing opponent policies or approximate expert. The first experiment only
checks whether lawful labels can be obtained. It does not fit a student.

## Information and rule contract

The original native expert receives the complete history projected to the
current actor: moves, public movement reveals and identities of dark pieces
captured by that actor. It receives no deal seed, actual hidden assignment,
future reveal, or opponent-private captured identity. A test poisons access to
the opponent's private capture identity and requires that it is never read.

Queries start from the original start position and replay every projected
token. Before accepting a root, the expert's echoed board, side, observer
inventory and no-capture counter must match the original learner observation.
Its complete legal action set must also match the source rule engine. Inventory
letter ordering is compared semantically. Each query clears the hash and
requests8192nodes with one thread,16MB hash and unchanged original it014 NNUE.
This requested node budget is not an exact completed-iteration count, a match
clock, or proof of the expert's strength under timed benchmark conditions.

A rejected history, mismatching root or different legal set quarantines the
whole episode. No accepted prefix of a failed episode is retained for training.
Resetting to a current FEN would erase repetition/chase history and is forbidden.
Source terminal outcomes remain Yuan-rule outcomes; they are not native expert
value labels, and final terminal equivalence is not established by this audit.

The expert supplies an actual selected move and, when available, the last
completed unbounded cp/mate score with its type and depth. The score is not
silently converted into a terminal-return target. A selected move is not an
MCTS visit distribution. No invented policy visit counts are recorded.

## Completed feasibility evidence and next data step

All eight prespecified original TRAIN episodes pass all368queried roots,
including15private dark captures. No episode is quarantined. Expert and
learner choices differ at286roots; disagreement alone does not establish
which action is better, student improvement or expert-label generalization.
All labels and source digests are archived. Three focused tests pass, covering
private-identity access, inventory comparison and a real native full-history
roundtrip with a private capture and rejected illegal history.

A separate replay-bound diagnostic avoids confusing sampled self-play actions
with search recommendations. The expert action is tied for maximum root visits
at178/368roots, but matches the sampled played move at only82/368. In the first
30plies, acting temperature is1and search also includes root noise. The expert
action has zero recorded visits at only3/368roots; its mean visit-policy mass
is0.12943. Thus this sample does not suggest that simply visiting more legal
actions addresses expert disagreement. No root in these eight games is fully
public, so no public-phase conclusion follows. These are reused training-root
agreement statistics, not regret or evidence that all expert choices are right.
Evidence: `results/observer-expert-policy-label-diagnostic-plan.json` and
`results/20261027-observer-expert-policy-label-diagnostic.json`.

The next frozen CPU-only annotation covers the original256TRAIN and32selection
episodes at the same budget. It includes these eight feasibility games, so it
is not an independent confirmation. Train/selection roles remain separated;
match layouts and external test games are absent. There is no new model fit
until compatibility and data availability are inspected. Any later training
must distinguish policy-label improvement from changes to the value target,
and must face the original warm actor and unchanged native teacher in fresh
paired matches. No public push or playing-strength claim follows from labels.

During the larger collection, game31first exposes a legal-set mismatch at
ply101: native permits `e4d4` and `e4e0`, which the source engine excludes.
The whole episode is quarantined. This is evidence that passing the small
feasibility sample cannot certify rule compatibility of the full corpus.
Even accepted roots establish agreement there, not equivalence of every
counterfactual continuation explored by the two engines. Expert annotations
and learner terminal returns must retain their respective rule provenance.

Evidence: `results/observer-expert-relabel-preflight-plan.json`,
`results/20261027-observer-expert-relabel-preflight.json` and
`results/observer-expert-relabel-corpus-plan.json`.

## Completed full annotation

All288prespecified source episodes are now processed. Eight complete episodes
are quarantined, all because the legal action sets differ: indices31,73,74,98,
149,184,246and257. The retained data contain249TRAIN games/13,304positions and
31selection games/1,677positions. No failed prefix is retained. In total15,571
roots were checked, of which14,981belong to accepted whole episodes; the extra
590checked roots are quarantined prefixes and are not training examples.

On the14,981accepted roots, the expert choice is unvisited in67cases, tied for
maximum visits in6,581cases, and selected by the noisy sampled actor in3,360.
Mean expert-action visit mass is0.13883. The286fully public roots come from20
accepted games; the expert is tied for maximum visits in116of them. Thus action
coverage alone does not explain the disagreement, including after physical
hidden pieces disappear. These are descriptive statistics on reused TRAIN and
selection trajectories, not independent evidence of policy quality or an
entropy-dependent causal effect.

All retained expert queries provide typed completed scores:14,489cp and492mate.
They remain separate from policy labels and source terminal returns. No critic
target or actor update is derived from them in this experiment. Label archives,
episode accounting and the follow-up diagnostic are recorded in
`results/20261027-observer-expert-relabel-corpus.json`,
`results/observer-expert-corpus-policy-diagnostic-plan.json` and
`results/20261027-observer-expert-corpus-policy-diagnostic.json`.

A subsequent training contrast must use identical accepted games for its
controls: the original256-game anchor cannot by itself control for the seven
TRAIN exclusions. Match teacher and learner supervision budgets explicitly,
keep selection games separate, and freeze fresh match seeds before fitting.
Changing policy labels and value targets simultaneously would not isolate
which change helped; retain separate controls if both are investigated.
