# Training-time privileged information: feasibility and research decision

Decision: investigate a **training-only privileged value estimator**, beginning
with supervised conditional-return prediction. Do not assume that direct
imitation of a clairvoyant move selector is an appropriate Jieqi objective.
No privileged evaluator or actual hidden assignment enters a deployed actor.

## Academic basis and qualifications

Asymmetric actor-critic permits different training inputs for actor and critic;
Pinto et al. demonstrate this empirically in robotics
([2017](https://arxiv.org/abs/1710.06542)). Baisero and Amato show why a
state-only critic can be biased in partially observable control and develop
history-state critics instead ([AAMAS2022](https://arxiv.org/html/2105.11674v2)).
Their analysis concerns the specified policy-gradient setting; it does not
automatically validate our current search-visit distillation pipeline.

Cai et al. identify failures of privileged expert-policy distillation, including
cases where unlimited imitation data does not recover the best partially
observable policy, and analyze belief-weighted privileged-value methods under
explicit assumptions ([NeurIPS2024, revised2025](https://arxiv.org/html/2412.00985v3)).
We have not established those assumptions for Jieqi. None of these papers
proves that granting our teacher dark identities improves student playing
strength, or that it is an original contribution by this project.

Jieqi is a two-player partially observable stochastic game; with the opponent
policy fixed, a player's learning problem can be viewed as a POMDP with suitable
state augmentation. Simultaneously changing both policies changes that problem.
Training/evaluation protocols must identify the generating policies rather than
silently importing fixed-environment guarantees into moving self-play.

## What should be learned

Let h be the actor's legal observation/action history, s the environment state
including hidden identities, and G the return under the **observer-limited
continuation policies**. The intended complete-information training target is

    V_priv(h,s) = E[G | h,s],
    V_actor(h) = E[V_priv(h,S) | h].

The second equality is conditional expectation, not an algorithm for obtaining
the correct posterior. It does not justify replacing a belief expectation with
the actual test deal, nor averaging clairvoyant best actions. An omniscient
optimal-policy value generally targets a different continuation policy.

For on-policy policy-gradient training, an action-independent, pre-action
baseline b(h,s) can exploit privileged information while the actor samples
actions only from h. This requires the relevant conditional independence;
bootstrapping an arbitrary state-only critic is a separate issue. Our current
learner instead fits search visits and terminal returns. Calling a replacement
value target an unbiased asymmetric actor-critic would be incorrect. An actual
policy-gradient extension must specify the acting distribution, search/policy
off-policy corrections, history representation and opponent policy treatment.

The first planned diagnostic estimates E[G | x,u], where x is the existing
history-omitting observer encoder and u contains live-dark identity labels.
That is a limited conditional predictor, **not** V_priv(h,s): neither the full
history nor both private capture ledgers are represented. It will use terminal
returns from the same observer-limited actor, so no full-information teacher
continuation is substituted into its labels.

## Why naive full-information action imitation can fail

Consider an illustrative two-stage hidden binary choice, not a Jieqi result.
Choosing the matching side gives+1 and the wrong side gives-1. A probe costs0.2
and reveals the state before the final choice. A clairvoyant teacher immediately
chooses the matching side and never probes. With equal hidden-state odds,
matching its one-hot actions from the aliased observation learns a50/50mixture:
expected return0. The partially observed policy can probe and obtain0.8.
With a fully revealing observation, the student can match the teacher and
obtain1. More teacher information is legitimate training supervision, but its
action preference need not teach the student to acquire missing information.

For Jieqi, retain face-down geometry even in a privileged model. Replacing a
dark square with an already revealed rook changes its movement rules and is
not merely supplying information. True identities belong in extra training
features, separate from the public board and legal move generator.

## Declining hidden information

Only live-dark identity privilege is considered initially. Its feature tensor
must become exactly zero when both sides have no live dark pieces. Phase
metrics should use live-dark count and, when justified, posterior uncertainty;
do not assume that unknown captured-piece ledger entropy decreases monotonically.
The public-only halving experiment exposed the fully public phase in only3/32
layout pairs, so delaying all intervention until that boundary is unlikely to
be an informative first test. Measure partial, transition and public strata
separately, without selecting favorable endpoints after results.

## Implemented data feasibility experiment

Some existing neural trajectories use lazy uniform without-replacement draws
when a dark piece is revealed/captured; they contain no ex ante assignment to
every dark square. Completing those assignments from the future realized game
could create hindsight correlations. The new training-only branch samples a
complete deal **before** choosing any actions, using a separate deal RNG.
It does not reinterpret old lazy episodes as recorded true layouts.

`research/privileged_training.py` keeps original dark-board geometry, validates
each remaining type count against the environment pool, and records90separate
square labels: true type on live darks, zero elsewhere. Before the existing
observer-limited PUCT receives a state clone, its actual assignment map is
removed. The original model, encoder, particles and search code are unchanged.
The module is not imported by any playing adapter.

Three targeted tests cover fixed-seed reproducibility, actual reveal/private
capture identities and remaining inventories; same-observation counterfactual
deals that change training labels but yield bitwise-identical actor search;
refusal to label lazy worlds and zero privilege after public collapse. These
tests do not prove all possible private-inventory legality paths are invariant.

The frozen eight-game,32-simulation warm-model GPU preflight completed480moves
and27private dark captures. Independent seeded replay checks the observation,
truth labels, move legality, visit-policy support, reveals and actor-relative
terminal targets both before and after writing safe NPZ arrays. All pass.
No game naturally reached a fully public position; that branch is unit-tested
only. These games are **data feasibility evidence**, with no fitted privileged
critic, student update, strong opponent or demonstrated chess improvement.
Plan and result: `results/privileged-data-preflight-plan.json` and
`results/20261027-privileged-data-preflight.json`; all eight arrays are archived.

## Next experimental decision

Before any actor change, freeze a paired value-prediction study: same games,
whole-game-separated training/selection/test splits, one training seed, identical
model capacity/initialization/minibatches/optimizer budget. The symmetric control
has zero extra channels; its privileged counterpart receives the live-dark
labels. Retain original observer features and report game-clustered uncertainty,
phase-stratified terminal-return error and calibration. Label shuffling is an
additional diagnostic, not a replacement for the symmetric control.

Fit on training games and choose checkpoints only on selection games. Audit
the fixed deployment path under counterfactual deals, test-only seed changes,
opponent-private ledger poisoning and removal of privileged artifacts. Never
input a game seed or future RNG state into either model. Do not reuse test
layouts as training examples. If conditional prediction fails to improve on
isolated test games, revise or stop this branch before a costly actor campaign.

Even lower critic error is insufficient: any later value distillation or actor
update must be tested using a frozen observer-only checkpoint in fresh paired
matches, with the existing native teacher attribution control. Only that can
support playing-strength progress. No public push is warranted by this preflight.

## Frozen paired value study has started

The four-worker corpus now generates352new complete fixed-deal games from the
same original warm observer-limited actor,128simulations andLeafBatch8.
Indices0–255are training,256–287selection and288–351final test, each with a
distinct seed beginning9601001. No preflight game is reused. Any failed or
resource-truncated game fails the corpus; no fabricated terminal labels or
silent replacement seeds are allowed. The corpus runner merges only completed
shards after checking all game counts, seeds and hashes.

Two250-channel value models retain the same original64-channel,15-block trunk
and value head. Twelve extra channels encode six live-dark types for each
color; the control sees zeros in those channels. Both retain exactly the same
capacity and original warm parameters, with all extra input weights initialized
to zero. Thus initial value functions agree exactly, while privileged channels
receive nonzero gradients after fitting. Unused policy parameters stay frozen;
all value/trunk parameters and their normalization statistics train normally.
No history planes are invented, so these remain conditional predictors rather
than full history-state critics.

Both profiles use one training seed20261027, eight epochs, batch128,
AdamWlr1e-5/weight decay0.0001, gradient clipping1 and the same game data,
minibatch permutations and horizontal-mirror draws. Each selects its minimum
whole-game mean selection MSE. **Both checkpoints are selected before test
arrays load.** The primary endpoint is symmetric-minus-privileged game-mean
terminal-return MSE on64test games, with10,000paired game-bootstrap resamples.
A positive95%lower bound is evidence for conditional predictive benefit only.
Position-weighted error, phase strata and calibration bins are descriptive
secondary outputs, without phase-specific superiority claims.

Three value-specific tests cover plane orientation, preserved observer inputs,
truth validation, mirror alignment, equal-capacity/exact initial functions,
zero-control versus nonzero privileged gradients, whole-game weighting and
suppression of tiny-sample positive claims. A real full-checkpoint CPU end-to-end
integration run fits six preflight games, selects on one and tests on one,
then safely reloads both selected critics; it is not a benefit estimate.

The frozen training controller waits for the verified live corpus process to
complete, then uses its released GPU. All source hashes are rechecked before
and after each stage. Completion will archive the two training curves, selected
checkpoint digests, held-out predictions and per-game uncertainty. The critic
contract and first-layer shape differ from ordinary playing actors; neither
artifact is deployable through the existing actor loader.

Plans: `results/privileged-value-{corpus,training,study}-plan.json`.
Integration: `results/20261027-privileged-value-training-integration.json`.
