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
