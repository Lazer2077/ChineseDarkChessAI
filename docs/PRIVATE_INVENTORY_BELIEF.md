# Action-conditioned private-inventory belief: exact core, unproven policy model

The original uniform root sampler marginalizes unknown captured piece types
without conditioning on the opponent's decisions. A possible extension is to
infer what the opponent knows from their actions. This is related to policy-
conditioned belief updates in imperfect-information games; see Brown et al.,
[ReBeL](https://arxiv.org/abs/2007.13544). We do not implement ReBeL's public-belief
solver, equilibrium guarantees or belief-value training. Neither Bayesian
conditioning nor exploiting fully revealed positions is claimed as novel.

## Finite count model

For one observer's own color, K counts public unrevealed types, including dark
pieces already captured privately by the opponent. Let d be their captured
count. The latent vector h satisfies sum(h)=d and 0<=h<=K. The live inventory is
c=K-h. The opponent knows h; the observer does not. Initially
P(h)=product_t binom(K_t,h_t)/binom(sum(K),d).

An opponent capture supplies no identity to the filter. It transitions h to
h+e_t with probability (K_t-h_t)/(sum(K)-d). An own public reveal t multiplies
each hypothesis by that probability and decrements K_t. An opponent decision a
multiplies by a declared likelihood pi(a|public,h). Action evidence precedes any
new private capture knowledge. Our own choices are not treated as opponent
evidence. Updates use normalized log weights, retain numerically underflowed
support and reject impossible observations without mutating the old state.

For an ordered future reveal sequence with type multiplicities k, its probability
is E_h[product_t (K_t-h_t)_{k_t}/(sum(K)-d)_{sum(k)}], using falling factorials.
Conditional ratios give without-replacement next-reveal probabilities. Under
the original uniform prior this recovers drawing from K, despite unseen
removals. Hidden-capture count support is bounded by the small fifteen-piece
inventory; no board-assignment enumeration is required.

The ledger entropy H(h) can increase after a hidden capture. The entropy of
remaining live assignments is H(h)+E_h log(N! / product_t c_t!). Both vanish
when that color has no live dark piece. Neither quantity has a pointwise
monotonic-decrease theorem under arbitrary observations. Both colors must have
zero live dark pieces before the whole board is public.

## Search and opponent-policy approximation

`research/private_inventory_search.py` changes root determinization only. The
root's observation, complete public rule history and original simulated private
observations are preserved. Uniform beliefs call the original sampler without
additional random draws. The original value/policy encoder does not encode the
posterior: this remains a limitation, not a belief-aware critic.

`research/private_inventory_policy.py` constructs each hypothetical opponent
view from public pools plus h. The observer's private captures of enemy pieces
are replaced by the enemy's public own pool. No actual opponent private ledger
is read. The frozen warm policy is normalized separately for each hypothesis
over the same public move mask, with temperature2 and epsilon0.1 uniform mixing.
It is a count-based approximation: capture order, full beliefs and full
strategic signaling are omitted. Exact Bayesian arithmetic is not an exact
opponent model or a Nash solution.

Eleven tests cover hypergeometric identities, informative action conditioning,
sequence ratios, impossible observations, rare-state recovery, full-information
collapse, unchanged RNG/search in the uniform control, complete rule-history
preservation, simulated opponent knowledge, private-ledger poisoning and proper
per-hypothesis policy normalization.

## First low-cost falsification

`results/private-inventory-calibration-plan.json` pins the first eight TRAIN
episodes of the original neural RL series, source/model digests and fixed
temperature/mixing parameters. No optimizer or benchmark seeds are used. Replay
checks all368 positions and both private ledgers. Actual hidden capture labels
are used only for scoring; never for opponent likelihood construction. The
public geometry mask is used only when no long-check/chase ban is active;
otherwise action conditioning is skipped because the upstream ban probe can
simulate reveals using a hidden pool. No decisions needed that skip here.

Across80 correlated ambiguous-inventory decisions, mean posterior log-score
gain over the uniform physical-update reference is -0.000343281 nats. Per-game
effects have both signs. This tiny reused-training diagnostic provides no
support for adopting the current raw-policy likelihood model. Its generating
actor used search, Dirichlet noise and action temperature, so likelihood-model
mismatch is a direct concern. No match or training campaign is launched for
this module. Improving and independently testing the opponent model would be
needed before a strength claim. Full records:
`results/20261026-private-inventory-calibration.json`.

## Stronger search-based likelihood: positive pilot fails to generalize

`research/native_opponent_policy.py` models the unchanged teacher's actions using
cold-hash fixed-depth2PVS scores for every public legal action under every h.
The per-hypothesis policy is softmax(score/200) mixed with10%uniform mass. Mate
scores use a declared signed distance mapping, not an exact inverse of UCI's
rounded mate notation. Each query uses one CPU thread and16MBHash. The actor's
own pool is public K; the observer's pool is K-h. Its true h is never supplied.
Searches use hypothetical FENs and thus reset check/chase/repetition history;
the original timed teacher also searches deeper. This remains a misspecified,
history-blind policy approximation even though the filter arithmetic is exact.

The first diagnostic uses the first four color-swapped pairs from old teacher-
versus-AB-JChess setA, sorted(pair,red), at most two earliest ambiguous teacher
decisions per game and support<=32. Both original source reports verify that
the generic candidate is the exact unchanged teacher binary+it014 network.
526public moves and all seeded private labels replay correctly;4,326forced
searches score14decisions. Mean cumulative log-score gain is+0.099636nats.
Retrospective last-scored-decision endpoints, assigning zero to games without
eligible decisions, average+0.092058nats/game across only four pairs. That small
pilot is dominated by one game; its bootstrap is not independent confirmation.

Before setB outcomes, a second fixed plan freezes the same model/depth/T/epsilon/
caps and a primary per-game endpoint: last-scored-decision logP(trueh) difference
versus the physical-update-only prior, zero if none eligible. Whole color-swapped
pairs, rather than repeated decisions, are the bootstrap units. It uses the
first twelve old setB pairs from a separate layout seed,24games,20,000bootstrap
replicates with seed20261102. These reused baseline trajectories are explicitly
development diagnostics for a new belief method; they are not fresh final
playing-strength confirmation data.

SetB checks1,685public moves and14,777forced searches, scoring44decisions. The
predeclared mean endpoint gain is **-0.036663nats/game**, pair-bootstrap95%
**[-0.199904,+0.059287]**. Mean cumulative decision gain is-0.003152nats.
The positive pilot does not generalize under this diagnostic. No live game or
training campaign is launched for this search-conditioned inventory filter.
Better opponent modeling or independently validated uncertainty calibration
would be required before spending larger compute or claiming a useful signal.

Four additional tests cover normalized/robust score policies, input inventory
provenance, exact-depth forced-search reproducibility after cold hash, and an
isolated full public replay whose frames remain unchanged when every private
captured-type field and setup seed are corrupted. Private labels are read only
after predictions/update in the actual scorer. No label-dependent frame/mask,
world pool or posterior likelihood is introduced.

Recipes: `results/native-inventory-calibration-plan.json` and
`results/native-inventory-calibration-b-plan.json`. Full predictions/labels and
endpoint analyses: `results/20261026-native-inventory-calibration-{a,b}{,-endpoints}.json`.
`scripts/analyze_inventory_calibration.py` computes the declared pair-level
endpoint analysis; setA endpoint analysis remains retrospective.
