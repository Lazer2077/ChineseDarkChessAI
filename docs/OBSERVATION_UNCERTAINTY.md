# A continuous observable measure of remaining hidden-type uncertainty

The public-only search switch reached its intended phase in only3/32development
layout pairs. A useful alternative should vary before every piece is revealed,
while distinguishing a player's certainty from publicly visible information.
This diagnostic prepares that measurement; no entropy-weighted model is trained.

## Declared finite-inventory measure

For one colour, K is the actor-known unrevealed inventory, including pieces
privately captured by the opponent. The latent removed count vector h has the
existing exchangeable hypergeometric prior, and n=sum(K)-sum(h)pieces remain
on board. Marginalizing unknown removed counts, the entropy of their ordered
type assignments is

    H_live = H(h) + E_h[ log(n! / product_t (K_t-h_t)!) ].

Given K and a complete live assignment, h is determined by conservation, so
this joint-entropy expression also describes the live assignment alone. Sum
the two colours and divide by the initial total39.29208748nats to obtain a
normalized stage measure. All inputs are public board/dark flags and the
current actor's known inventory; score, outcome, future moves, true assignments,
seeds and the opponent's private ledger are excluded.

This is exact for the **declared count-only prior**, not a full adversarial
posterior conditioned on an opponent's action-selection likelihood, and not an
entropy of every possible opponent belief/history. It measures physical live
types, rather than the entire informational state of the stochastic game.
Do not infer that every individual entropy must strictly decrease after every
observation; the number of live pieces and private-ledger uncertainty describe
different quantities.

H_live=0may occur with dark pieces still on board when their types are already
inferable. They still obey face-down movement geometry, and the opponent need
not share this certainty. **Do not activate a publicly revealed state gate**
from that condition. The original both-dark-zero public gate is unchanged.

## Training-only diagnostic

All256original first-RL training games/13,924positions are audited before use.
The same observer-compatible legal masks bind the completed-visit targets.
Three targeted tests cover an exact finite-urn sequence entropy, real initial
deal/label isolation, distinct type certainty/public reveal and invalid counts.

There are267fully public positions plus352type-certain positions with darks
still present. Mean normalized entropy is0.36947027. Descriptive target entropy
is measured from actual normalized visits, without using terminal labels:

| Normalized live entropy | Positions | Mean live darks | Mean target entropy | Target entropy/log(legal actions) |
| --- | ---: | ---: | ---: | ---: |
| Type-certain | 619 | 0.706 | 2.3281 | 0.7923 |
| (0,0.25] | 5,108 | 6.110 | 2.8322 | 0.8558 |
| (0.25,0.5] | 3,844 | 14.189 | 3.0929 | 0.8976 |
| (0.5,0.75] | 2,596 | 20.615 | 3.1551 | 0.9060 |
| (0.75,1] | 1,757 | 27.123 | 3.3202 | 0.9077 |

Correlation with search-target entropy is0.26797; normalizing target entropy
by the legal-set maximum reduces it to0.14360. Branching, exploration noise,
phase and trajectory selection remain confounded. Broad visit targets can
represent multiple good moves; this does **not** measure conditional return
variance or prove that a high-entropy target is worse supervision.

## Candidate future control, not an adopted recipe

An observable-dependent teacher coefficient H_live/mean_train(H_live)would
have mean1on these training records, the same average coefficient as the
current constant-unit anchor. Its recorded range is[0,2.70657774]: strongest
early, zero at type certainty/public collapse. This is one testable way to
retain more teacher guidance in uncertain states and loosen it as types become
known. Equal mean coefficients do not ensure equal realized losses, gradient
norms or effective learning strength.

The candidate weights are archived solely for reproducibility. No new training
coefficient, checkpoint or test set is chosen from this descriptive study, and
no playing-strength improvement is implied. Finish the existing constant-anchor
campaign before selecting a new controlled ablation.

Evidence: `results/observation-uncertainty-diagnostic-plan.json` and
`results/20261027-observation-uncertainty-diagnostic.json`, with safe per-record
entropy/weight arrays and source/data digests.

## Controlled implementation prepared, no new campaign selected

`research/entropy_policy_anchor.py` computes the per-record legal teacher KL
before applying the frozen observable weights. Loss is the **batch mean of
weighted KL**, without dividing by each minibatch's weight sum. Otherwise a
second normalization would change the specified whole-corpus objective. Weight
gradients are stopped; zero-weight records receive zero auxiliary gradient.
All-ones mode delegates to the established constant loss exactly.

The loader binds episode ordering and hashes, safely loads arrays, then
independently recomputes entropy from permitted observer-record bytes. It
rejects a mismatched coefficient vector even if its global mean and file hash
are internally valid. Mirroring changes spatial inputs/masks but leaves these
count-entropy weights invariant. It introduces no new inference input or actor.

`scripts/train_entropy_anchored_selfplay.py` extends the existing guarded
training loop without modifying any file pinned by running experiments.
Teacher queries, original CE/terminal MSE, normalization, augmentation, optimizer
and checkpoint-selection criterion remain as in the constant-anchor wrapper.
The loop records its weighting mode/range/mean and explicitly labels the logged
auxiliary loss as **weighted** KL. This has no adaptive critic, population-based
schedule or new policy-gradient guarantee.

Three tests verify exact constant-loss/gradient reproduction, analytic weighted
gradients including a zero-weight row, detached teacher/weights, rejection of
negative coefficients and observer-only recomputation/row-order checks. Poisoned
score/outcome/move bytes cannot change the loaded coefficients.

CPU integration uses the same four old TRAIN games/186positions plus two other
TRAIN games for smoke selection, one epoch/three updates. Constant mode exactly
reproduces the prior unit-anchor selected arrays
`0dca3bd68d70e9720bd362c56495001d7651ecd0dd04b69b63509aed7667c74a`.
The variable-weight mode stays finite, has mean1and range[0.1421473,2.2476754]
on **this smoke subset**, and changes weights as intended. This is not the
full-cohort coefficient range, a new selected research recipe or playing-strength
evidence. Finish the existing constant-anchor campaign before any larger fit.
Evidence: `results/20261027-entropy-anchor-integration.json`.
